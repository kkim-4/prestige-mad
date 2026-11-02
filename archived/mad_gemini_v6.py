#!/usr/bin/env python3
"""
mad_gemini_v5.py — prestige-weighted, confidence-scored multi-agent debate, one file.

v5 adds
  * --backend openai: any OpenAI-compatible server, e.g. open models on a GPU with vLLM
      vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
      python3 mad_gemini_v5.py trials --backend openai --model Qwen/Qwen2.5-7B-Instruct ...
    (OPENAI_BASE_URL defaults to http://localhost:8000/v1; OPENAI_API_KEY defaults to EMPTY)
  * --provider: hosted or local open models without your own GPU (implies --backend openai)
      ollama        local Ollama            http://localhost:11434/v1      (no key)
      ollama_cloud  Ollama cloud models     https://ollama.com/v1          OLLAMA_API_KEY
      openrouter    OpenRouter              https://openrouter.ai/api/v1   OPENROUTER_API_KEY
      groq          Groq                    https://api.groq.com/openai/v1 GROQ_API_KEY
      together      Together AI             https://api.together.xyz/v1    TOGETHER_API_KEY
      hf            Hugging Face providers  https://router.huggingface.co/v1  HF_TOKEN
      deepinfra     DeepInfra               https://api.deepinfra.com/v1/openai  DEEPINFRA_API_KEY
      vllm          your own vLLM server    http://localhost:8000/v1       (no key)
    Free tiers have strict rate and daily limits: pace calls with --stagger (60 / requests-per-minute)
    and use --max-retries for longer waits after "429 Too Many Requests".
  * mechanism ablations of dual (same majority vote and later-round rule as dual):
      seq_dual       leader after round 1, dissenter after round 2   (does simultaneity matter?)
      diss_only      only the top dissenter after round 1            (is the minority's case enough?)
      lead_rand      leader + a random other agent, any answer       (does it need a dissenter?)
      dual_randdiss  leader + a random agent with a different answer (does choosing the dissenter matter?)
    Random picks are seeded by the problem and the agent names, so reordering agents
    does not change them.

v4 adds `pool_dual`: round 1 already produces 3 independent samples per agent (15 in all).
Every answer that appears in ANY sample is a candidate, even one no agent ended up choosing.
Each candidate gets a support score  sum over agents of  prestige x (share of that agent's
samples giving it) x (its stated confidence in them)  -- a prestige-weighted opinion pool.
After round 1 the best-supported argument for the top TWO candidates is shown (taken from
the sample that argued it best), so a right answer an agent only half-believed can still be
heard. The debate runs whenever the samples disagree, not only the agents' top answers.
Needs fresh round-1 answers (sample rationales are now stored), so don't use --init-from
with a store made by an older version.

v3 adds the `dual` policy (leader and strongest dissenter speak together after round 1,
then any unheard answer; one-agent-one-vote final vote with prestige x confidence as
tie-break) and its same-airtime rotation control `rr_dual`. Also stores every agent's
answer per round (for round_metrics.py) and supports --warmup-from / --warmup-n.

Setup:
    pip install google-genai pydantic
    export GEMINI_API_KEY="..."

Use:
    python3 mad_gemini.py smoke                     # 1 call per transport, full errors
    python3 mad_gemini.py mock                      # mechanism sanity check, no API
    python3 mad_gemini.py calib --n 20              # does stated confidence track correctness?
    python3 mad_gemini.py debate --problems math.jsonl --n 15
    python3 mad_gemini.py trials --problems math.jsonl --n 100 --seeds 3 --workers 6 --stagger 0.2
    python3 mad_gemini.py report --store trials.jsonl       # regrade + summarize, no API calls

Transport: defaults to the stable generateContent endpoint. Use --api interactions
to try the newer one.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field

# ===========================================================================
# 1. SCHEMA — the contract every agent response must satisfy
# ===========================================================================

class AgentTurn(BaseModel):
    rationale: str = Field(description="Brief reasoning, at most 3 sentences.")
    answer: str = Field(description="Final answer only, no units or explanation.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Honest probability that `answer` is correct. Scored with a proper "
                    "scoring rule against ground truth after the problem closes.")
    key_uncertainty: str = Field(
        description="The single thing most likely to make `answer` wrong.")

    def clipped(self) -> "AgentTurn":
        self.confidence = min(0.99, max(0.01, float(self.confidence)))
        self.answer = str(self.answer).strip()
        return self


# ===========================================================================
# 2. SCORING — proper scoring rules, prestige ledger, calibration diagnostics
# ===========================================================================

def brier(c: float, correct: bool) -> float:
    """Rescaled Brier in [0,1], higher better. c=1 & wrong -> 0; c=0.5 & right -> 0.75."""
    return 1.0 - (c - (1.0 if correct else 0.0)) ** 2


def log_score(c: float, correct: bool) -> float:
    """Log score mapped to (0,1]; punishes overconfidence much harder than Brier."""
    return max(1e-6, c if correct else 1.0 - c)


RULES = {"brier": brier, "log": log_score}


@dataclass
class PrestigeLedger:
    """Carried-forward, socially visible record of each agent's calibration.

    Updated ONLY after a problem closes, so nothing about correctness leaks
    into the debate itself.
    """
    agent_ids: list
    rule: str = "brier"
    alpha: float = 0.3                   # EMA rate
    prior: float = 0.5                   # prestige of an agent with no history
    stake_by_confidence: bool = False    # high stated confidence risks more prestige
    frozen: bool = False
    values: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)   # agent -> [(conf, correct, score)]

    def __post_init__(self):
        for a in self.agent_ids:
            self.values.setdefault(a, self.prior)
            self.records.setdefault(a, [])

    def get(self, a: str) -> float:
        return self.values.get(a, self.prior)

    def update(self, a: str, confidence: float, correct: bool) -> float:
        s = RULES[self.rule](confidence, correct)
        self.records.setdefault(a, []).append((confidence, correct, s))
        if self.frozen:
            return self.get(a)
        rate = min(1.0, self.alpha * (0.5 + confidence)) if self.stake_by_confidence else self.alpha
        self.values[a] = (1 - rate) * self.get(a) + rate * s
        return self.values[a]

    def snapshot(self) -> dict:
        return dict(self.values)

    def mean_score(self, a: str) -> float:
        r = self.records.get(a, [])
        return sum(s for _, _, s in r) / len(r) if r else float("nan")

    def ece(self, a: str, bins: int = 5) -> float:
        """Expected calibration error: gap between stated confidence and accuracy."""
        r = self.records.get(a, [])
        if not r:
            return float("nan")
        buckets = [[] for _ in range(bins)]
        for c, y, _ in r:
            buckets[min(bins - 1, int(c * bins))].append((c, y))
        err = 0.0
        for b in buckets:
            if b:
                conf = sum(c for c, _ in b) / len(b)
                acc = sum(1 for _, y in b if y) / len(b)
                err += len(b) / len(r) * abs(conf - acc)
        return err


@dataclass
class PlattScaler:
    """p = sigmoid(a*logit(c)+b), fitted by gradient descent.

    The post-hoc calibration baseline: confidence is corrected after the fact
    but never *costs* the agent anything.
    """
    a: float = 1.0
    b: float = 0.0

    @staticmethod
    def _logit(c: float) -> float:
        c = min(0.999, max(0.001, c))
        return math.log(c / (1 - c))

    def fit(self, pairs, steps: int = 500, lr: float = 0.05):
        if len(pairs) < 5:
            return self
        xs = [self._logit(c) for c, _ in pairs]
        ys = [1.0 if y else 0.0 for _, y in pairs]
        for _ in range(steps):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = 1 / (1 + math.exp(-(self.a * x + self.b)))
                ga += (p - y) * x
                gb += (p - y)
            self.a -= lr * ga / len(xs)
            self.b -= lr * gb / len(xs)
        return self

    def __call__(self, c: float) -> float:
        return 1 / (1 + math.exp(-(self.a * self._logit(c) + self.b)))


# ===========================================================================
# 3. AGENTS — interface: propose(problem, transcript, prestige) -> AgentTurn
# ===========================================================================

PERSONAS = {
    "careful":    "You are a careful, methodical solver. Verify each step before committing.",
    "fast":       "You are a fast, intuitive solver. Commit quickly, but report confidence honestly.",
    "skeptic":    "You are a skeptic. Before agreeing with anyone, look for the flaw in their reasoning.",
    "teacher":    "You are a patient teacher. Reason cleanly and simply.",
    "contrarian": "You try an alternative approach before accepting the obvious one.",
}

SYSTEM = """{persona}

You are one agent in a small group solving a problem over several rounds.
Each round you see the transcript of who spoke, what they answered, their
stated confidence, and their PRESTIGE (0-1): a running record of how well
that agent's stated confidence has matched actual correctness on past problems.

Your own stated confidence will be scored against the true answer with a
proper scoring rule after this problem closes. Being confident and wrong
costs you prestige; being right but hedged also costs you some. You maximise
your expected score only by reporting the probability you actually believe.
Prestige decides how much say you get in future rounds.

Do not defer to another agent merely because they sound sure. Weigh their
prestige, their reasoning, and your own work."""

USER = """PROBLEM:
{question}

CURRENT PRESTIGE: {standings}

TRANSCRIPT SO FAR:
{board}

Solve the problem and respond in the required JSON format."""


class RateLimiter:
    """Paces the START of every API call across all agents/threads so a
    round's 5 simultaneous requests become a staggered burst instead of
    landing on the server in the same instant. Threads block in FIFO-ish
    order until min_interval has elapsed since the last call started."""
    def __init__(self, min_interval: float = 0.3):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            start_at = max(now, self._next_ok)
            self._next_ok = start_at + self.min_interval
            delay = start_at - now
        if delay > 0:
            time.sleep(delay)


_RATE_LIMITER = RateLimiter(float(os.environ.get("MAD_STAGGER", "0.3")))


@dataclass
class GeminiAgent:
    agent_id: str
    persona: str
    model: str = None
    temperature: float = 0.7
    thinking_level: str = None          # "low" | "high"; None = model default
    log_path: str = "calls.jsonl"
    max_retries: int = 5
    api: str = None                     # "legacy" (default) | "interactions"; env MAD_API
    verbose_errors: bool = True

    _lock = threading.Lock()

    def __post_init__(self):
        from google import genai
        from google.genai import types
        self.model = self.model or os.environ.get("MAD_MODEL", "gemini-3.8-flash")
        self.api = self.api or os.environ.get("MAD_API", "legacy")
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")
        if not use_vertex and not os.environ.get("GEMINI_API_KEY"):
            sys.exit("Neither GEMINI_API_KEY nor GOOGLE_GENAI_USE_VERTEXAI is set.\n"
                     "AI Studio: export GEMINI_API_KEY='...'\n"
                     "Vertex AI: export GOOGLE_GENAI_USE_VERTEXAI=True "
                     "GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=us-central1")
        # 120 s timeout so slow thinking responses don't get dropped
        self.client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        self.system = SYSTEM.format(persona=PERSONAS[self.persona])

    def propose(self, problem: dict, transcript: list, prestige: dict) -> AgentTurn:
        board = "\n".join(
            f"[round {t['round']}] {t['speaker']} (prestige {t['speaker_prestige']:.2f}, "
            f"confidence {t['confidence']:.2f}) -> {t['answer']}\n   reasoning: {t['rationale']}"
            for t in transcript
        ) or "(no one has spoken yet)"
        standings = ("(hidden in the first round)" if prestige is None else
                     ", ".join(f"{a}: {p:.2f}" for a, p in sorted(prestige.items())))
        prompt = USER.format(question=problem["question"], standings=standings, board=board)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                _RATE_LIMITER.wait()
                t0 = time.time()
                text = self._call(prompt)
                turn = AgentTurn.model_validate_json(text).clipped()
                self._log({"agent": self.agent_id, "model": self.model, "api": self.api,
                           "ts": time.time(), "q": problem["question"], "round": len(transcript),
                           "out": turn.model_dump(), "latency_s": round(time.time() - t0, 2)})
                return turn
            except Exception as e:
                last_err = e
                if self.verbose_errors:
                    print(f"  [{self.agent_id}] attempt {attempt+1}/{self.max_retries}: "
                          f"{type(e).__name__}: {str(e)[:140]}", flush=True)
                time.sleep(2 ** attempt + random.uniform(0, 1))   # 1-2, 2-3, 4-5, 8-9, 16-17 s
        raise RuntimeError(f"{self.agent_id}: failed after {self.max_retries} tries: {last_err}")

    def _call(self, prompt: str) -> str:
        """Returns raw JSON text. Two transports, same schema enforcement."""
        from google.genai import types
        if self.api == "interactions":
            gen = {"temperature": self.temperature}
            if self.thinking_level:
                gen["thinking_level"] = self.thinking_level
            it = self.client.interactions.create(
                model=self.model, input=prompt, system_instruction=self.system,
                generation_config=gen, store=False,
                response_format={"type": "text", "mime_type": "application/json",
                                 "schema": AgentTurn.model_json_schema()})
            return it.output_text
        cfg = types.GenerateContentConfig(
            system_instruction=self.system,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=AgentTurn,
        )
        if self.thinking_level:
            cfg.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)
        return self.client.models.generate_content(
            model=self.model, contents=prompt, config=cfg).text

    def _log(self, rec: dict):
        if not self.log_path:
            return
        with self._lock, open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


PROVIDERS = {   # name: (base URL, environment variable holding the key, or None)
    "vllm": ("http://localhost:8000/v1", None),
    "ollama": ("http://localhost:11434/v1", None),
    "ollama_cloud": ("https://ollama.com/v1", "OLLAMA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}


def _json_object(text: str) -> str:
    """Pull the JSON object out of a reply (strips <think>...</think> and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i >= 0 and j > i else text


def _normalize_turn(text: str) -> str:
    """Open models often return the answer as a number, confidence as "90%" or a string,
    or leave a field out. Coerce to the AgentTurn shape; the content is never changed."""
    # LaTeX inside JSON strings: "\\frac" is fine, but a bare "\frac" either breaks the JSON or
    # silently becomes a form feed + "rac". Keep only \" \\ \/ \uXXXX as escapes; every other
    # backslash is kept literally, so LaTeX answers survive intact.
    fix = lambda m: m.group(0) if (m.group(1) in ('"', "\\", "/") or m.group(1).startswith("u")) \
        else "\\\\" + m.group(1)
    text = re.sub(r'\\(u[0-9a-fA-F]{4}|["\\/]|.)', fix, text, flags=re.S)
    try:
        d = json.loads(text, strict=False)         # strict=False: raw newlines inside strings are OK
    except Exception:
        return text                               # let validation report it (the call is retried)
    if not isinstance(d, dict):
        return text
    d = {k.lower().strip(): v for k, v in d.items()}
    ans = d.get("answer", d.get("final_answer", ""))
    d["answer"] = ans if isinstance(ans, str) else json.dumps(ans) if isinstance(ans, (list, dict)) else str(ans)
    c = d.get("confidence", 0.5)
    try:
        c = float(str(c).strip().rstrip("%"))
        c = c / 100 if c > 1 else c
    except ValueError:
        c = 0.5
    d["confidence"] = min(1.0, max(0.0, c))
    for k in ("rationale", "key_uncertainty"):
        v = d.get(k, "")
        d[k] = v if isinstance(v, str) else json.dumps(v)
    return json.dumps({k: d[k] for k in ("rationale", "answer", "confidence", "key_uncertainty")})


@dataclass
class OpenAIAgent(GeminiAgent):
    """Same agent, any OpenAI-compatible chat server (vLLM, SGLang, llama.cpp, Ollama, ...).
    Structured output is requested as a JSON schema; the reply is also parsed leniently."""
    base_url: str = None
    max_tokens: int = 1024
    extra_body: dict = None
    schema_mode: str = "json_schema"      # json_schema | json_object | guided_json | none
    api_key: str = None

    _modes_seen = {}                      # base_url -> schema mode that the server accepted

    def __post_init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("--backend openai needs the openai package: pip install openai")
        self.api = "openai"
        self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=self.base_url,
                             api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                             timeout=300, max_retries=0)
        self.system = SYSTEM.format(persona=PERSONAS[self.persona])

    def _call(self, prompt: str) -> str:
        """Asks for JSON in the strictest form the server accepts. If a server rejects JSON-schema
        output (many hosted free tiers do), it falls back to json_object, then to plain text, and
        remembers the mode that worked for this server."""
        import openai
        schema = AgentTurn.model_json_schema()
        system = self.system + ("\n\nReply with one JSON object with the keys rationale, answer, "
                                "confidence (0-1) and key_uncertainty, and nothing else. Work the problem "
                                "out in your head first. The rationale must be at most 3 short sentences "
                                "(under 60 words) and key_uncertainty one short phrase; no tables, no "
                                "step-by-step working in the JSON.")
        order = ["json_schema", "json_object", "none"] if self.schema_mode == "json_schema" else [self.schema_mode]
        start = self._modes_seen.get(self.base_url)
        if start in order:
            order = order[order.index(start):]
        last = None
        for mode in order:
            kw, extra = {}, dict(self.extra_body or {})
            if mode == "json_schema":
                kw["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "AgentTurn", "schema": schema}}
            elif mode == "json_object":
                kw["response_format"] = {"type": "json_object"}
            elif mode == "guided_json":
                extra["guided_json"] = schema
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    extra_body=extra or None, **kw)
            except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
                last = e                         # format not supported here: try the next one
                continue
            self._modes_seen[self.base_url] = mode
            return _normalize_turn(_json_object(r.choices[0].message.content or ""))
        raise last


@dataclass
class MockAgent:
    """Simulated agent with known accuracy, confidence bias and conformity.
    Lets you test the mechanism with zero API calls."""
    agent_id: str
    accuracy: float
    conf_bias: float = 0.0
    conformity: float = 0.5
    rng: random.Random = None

    def __post_init__(self):
        self.rng = self.rng or random.Random(hash(self.agent_id) & 0xFFFF)

    def propose(self, problem, transcript, prestige) -> AgentTurn:
        correct = self.rng.random() < self.accuracy
        d = problem.get("distractors") or [str(problem["answer"]) + "0", "-1"]
        answer = problem["answer"] if correct else (d[0] if self.rng.random() < 0.6 else self.rng.choice(d))
        p = self.accuracy if correct else 1 - self.accuracy
        if transcript:
            last = transcript[-1]
            if self.rng.random() < self.conformity * last["speaker_prestige"] * last["confidence"]:
                answer, p = last["answer"], max(p, last["confidence"] * 0.8)
        c = min(0.99, max(0.01, p + self.conf_bias + self.rng.gauss(0, 0.05)))
        return AgentTurn(rationale="mock", answer=str(answer), confidence=c, key_uncertainty="")


# ===========================================================================
# 4. ORCHESTRATOR — one problem = one debate = N rounds
# ===========================================================================

POLICIES = ["prestige_x_conf", "conf_only", "posthoc", "prestige_only", "round_robin", "fixed", "both_sides",
            "dual", "rr_dual", "pool_dual", "seq_dual", "diss_only", "lead_rand", "dual_randdiss"]
ABLATIONS = ("seq_dual", "diss_only", "lead_rand", "dual_randdiss")
MULTI = ("dual", "rr_dual", "pool_dual") + ABLATIONS        # policies that use _pick_many
MAJORITY = ("dual", "pool_dual") + ABLATIONS                # one agent, one vote; prestige x conf breaks ties


def _wrap(x: str) -> str:
    """Parenthesize a fraction part unless it's a single atom, so
    \\frac{11+9a}{20} becomes (11+9a)/20 rather than 11+9a/20."""
    x = x.strip()
    return x if re.fullmatch(r"[\w.]+", x) else f"({x})"


_FRAC_ARG = r"(\{[^{}]+\}|\w)"


def _strip_latex(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    # vectors/matrices -> tuple: \begin{pmatrix} a \\ b \end{pmatrix} -> (a,b)
    s = re.sub(r"\\begin\{[pbv]?matrix\}(.*?)\\end\{[pbv]?matrix\}",
               lambda m: "(" + ",".join(x.strip() for x in re.split(r"\\\\|&", m[1])) + ")", s)
    # \frac with or without braces: \frac{a}{b}, \frac 34, \frac43, \frac{3}4
    s = re.sub(r"\\frac\s*" + _FRAC_ARG + r"\s*" + _FRAC_ARG,
               lambda m: _wrap(m[1].strip("{}")) + "/" + _wrap(m[2].strip("{}")), s)
    s = re.sub(r"\\sqrt\s*" + _FRAC_ARG, lambda m: f"sqrt({m[1].strip('{}')})", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)            # 4343_{6} -> 4343_6
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("$", "").replace("\\!", "").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\", "")
    s = s.strip().replace(" ", "")
    return s


def norm(ans) -> str:
    return _strip_latex(ans).lower().replace(",", "").rstrip(".")


def _sympy_equal(na: str, nb: str) -> bool:
    # skip plain words: implicit multiplication would make "no" == "on"
    if re.fullmatch(r"[a-z]+", na) or re.fullmatch(r"[a-z]+", nb):
        return False
    try:
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application)
        import sympy
        tr = standard_transformations + (implicit_multiplication_application,)
        ea = parse_expr(na.replace("^", "**"), transformations=tr)
        eb = parse_expr(nb.replace("^", "**"), transformations=tr)
        return sympy.simplify(ea - eb) == 0
    except Exception:
        return False


def answers_match(a, b) -> bool:
    """String match after LaTeX normalization; falls back to sympy for
    algebraic/numeric equivalence (e.g. 14/3 == \\frac{14}{3} == 4.6667,
    (11+9a)/20 == \\frac{9a+11}{20}). Base subscripts are optional (4343_6 == 4343)."""
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    # multiple choice (e.g. GPQA): truth is a single letter; accept "B", "(B)", "B)", "B. text"
    for x, y in ((na, nb), (nb, na)):
        if re.fullmatch(r"[a-e]", y):
            m = re.match(r"\(?([a-e])(\)|\.|:|$)", x)
            if m:
                return m.group(1) == y
            if re.fullmatch(r"(option|answer)?\(?[a-e]\)?", x):
                return x.strip("()")[-1] == y
    if _sympy_equal(na, nb):
        return True
    # a base subscript on only one side: compare without it
    sa, sb = re.sub(r"_\d+$", "", na), re.sub(r"_\d+$", "", nb)
    if (sa, sb) != (na, nb) and sa == sb:
        return True
    return False


@dataclass
class DebateResult:
    final_answer: str
    correct: bool
    transcript: list
    submissions: dict
    prestige_before: dict
    prestige_after: dict = field(default_factory=dict)


class Orchestrator:
    """Each round: everyone proposes in parallel -> the policy picks ONE speaker
    -> that turn is appended to the shared transcript. After the last round the
    final answer is a weighted vote over each agent's last submission."""

    def __init__(self, agents, ledger: PrestigeLedger, policy="prestige_x_conf", rounds=3,
                 tau=None, scaler: PlattScaler = None, parallel=True, seed=0, random_ties=False,
                 stop_unanimous=False, carry=None, pool=None):
        self.agents, self.ledger, self.policy, self.rounds = agents, ledger, policy, rounds
        self.random_ties = random_ties       # True: equal weights -> random speaker, not list order
        self.stop_unanimous = stop_unanimous # True: end the debate as soon as every agent agrees
        self.carry = carry                   # {agent: [its round-1 sample answers]} -> later-round confidence
        self.pool = pool                     # {agent: [round-1 sample dicts]} -> pool_dual candidates
        self.tau = tau                       # None -> argmax speaker; float -> softmax
        self.scaler = scaler or PlattScaler()
        self.parallel = parallel
        self.rng = random.Random(seed)
        self._seed_base = str(seed).rsplit(":", 1)[0]   # without the canon/order tag

    def weight(self, agent_id: str, turn: AgentTurn) -> float:
        p, c = self.ledger.get(agent_id), turn.confidence
        return {"prestige_x_conf": p * c, "fixed": p * c, "conf_only": c,
                "posthoc": self.scaler(c), "prestige_only": p, "round_robin": 1.0,
                "both_sides": p * c, "dual": p * c, "rr_dual": 1.0,
                "pool_dual": p * c, "seq_dual": p * c, "diss_only": p * c,
                "lead_rand": p * c, "dual_randdiss": p * c}[self.policy]

    def _best(self, turns):
        ws = [self.weight(a, t) for a, t in turns]
        m = max(ws)
        tied = [x for x, w in zip(turns, ws) if w >= m - 1e-12]
        return self.rng.choice(tied) if self.random_ties else tied[0]

    def _pool_options(self):
        """Every answer in any round-1 sample, with its prestige-weighted support and its
        best argument (the sample whose agent has the highest prestige x share x stated conf)."""
        opts = []                                # [answer, support, (score, agent, sample, share)]
        for aid, ss in self.pool.items():
            pr = self.ledger.get(aid)
            for x in ss:
                share = sum(answers_match(y["answer"], x["answer"]) for y in ss) / len(ss)
                sc = pr * share * x["confidence"]
                for o in opts:
                    if answers_match(o[0], x["answer"]):
                        break
                else:
                    o = [x["answer"], 0.0, None]
                    opts.append(o)
                if o[2] is None or sc > o[2][0]:
                    o[2] = (sc, aid, x, share)
        for o in opts:                           # support: once per agent per option
            o[1] = sum(self.ledger.get(aid) * (sum(answers_match(y["answer"], o[0]) for y in ss) / len(ss)) *
                       max([y["confidence"] for y in ss if answers_match(y["answer"], o[0])], default=0)
                       for aid, ss in self.pool.items())
        return sorted(opts, key=lambda o: -o[1])

    def _as_turn(self, best):
        sc, aid, x, share = best
        return aid, AgentTurn(rationale=x["rationale"], answer=x["answer"],
                              confidence=round((share + x["confidence"]) / 2, 4), key_uncertainty="")

    def _stable_choice(self, cands, rnd):
        """Random pick that ignores agent order: same problem -> same pick in the reordered run."""
        cands = sorted(cands, key=lambda x: x[0])
        return random.Random(f"{self._seed_base}:{rnd}:{self.policy}").choice(cands)

    def _pick_many(self, turns, rnd):
        """Speakers for 'dual' and its same-airtime control 'rr_dual'.
        dual, after round 1: the top prestige x confidence agent AND the top agent holding a
          different answer speak together, so everyone hears both sides before revising.
        dual, later rounds: the top agent holding an answer nobody has argued yet; if every
          answer has been heard, the top agent overall.
        rr_dual: same number of voices (2, then 1), chosen by rotation instead."""
        if self.policy == "pool_dual" and self.pool:
            opts = self._pool_options()
            if rnd == 0:
                return [self._as_turn(o[2]) for o in opts[:2]]
            heard = lambda ans: any(answers_match(ans, h) for h in self._spoken)
            live = [x for x in turns if not heard(x[1].answer)]
            if live:                             # someone now holds an answer nobody has argued
                return [self._best(live)]
            hidden = [o for o in opts if not heard(o[0])]
            if hidden:                           # best-supported sample answer not yet argued
                return [self._as_turn(hidden[0][2])]
            return [self._best(turns)]
        if self.policy == "rr_dual":
            k, n = (2 if rnd == 0 else 1), len(turns)
            out = [turns[(self._rr_next + j) % n] for j in range(k)]
            self._rr_next += k
            return out
        if self.policy in ABLATIONS and rnd == 0:
            lead = self._best(turns)
            diss = [x for x in turns if not answers_match(x[1].answer, lead[1].answer)]
            if self.policy == "seq_dual":       # dissenter comes next round (via the unheard rule)
                return [lead]
            if self.policy == "diss_only":
                return [self._best(diss)] if diss else [lead]
            if self.policy == "lead_rand":
                return [lead, self._stable_choice([x for x in turns if x[0] != lead[0]], rnd)]
            if self.policy == "dual_randdiss":
                return [lead] + ([self._stable_choice(diss, rnd)] if diss else [])
        if rnd == 0:
            lead = self._best(turns)
            diss = [x for x in turns if not answers_match(x[1].answer, lead[1].answer)]
            return [lead] + ([self._best(diss)] if diss else [])
        unheard = [x for x in turns if not any(answers_match(x[1].answer, h) for h in self._spoken)]
        return [self._best(unheard or turns)]

    def _pick(self, turns, rnd):
        if self.policy == "round_robin":
            return turns[rnd % len(turns)]
        if self.policy == "both_sides" and getattr(self, "_spoken", None):
            # After the top-ranked agent has spoken, the floor goes to the top-ranked
            # agent holding an answer nobody has argued yet, so a confidently wrong
            # leader is always challenged by the most credible alternative.
            fresh = [(a, t) for a, t in turns if not any(answers_match(t.answer, x) for x in self._spoken)]
            if fresh:
                turns = fresh
        ws = [self.weight(a, t) for a, t in turns]
        if self.tau is None:
            if self.random_ties:
                best = max(ws)
                return self.rng.choice([t for t, w in zip(turns, ws) if w >= best - 1e-12])
            return turns[max(range(len(ws)), key=ws.__getitem__)]
        return self.rng.choices(turns, weights=[math.exp(w / self.tau) for w in ws], k=1)[0]

    def run(self, problem: dict, update_prestige=True, round0: dict = None) -> DebateResult:
        transcript, subs = [], defaultdict(list)
        before = view = self.ledger.snapshot()
        self._rr_next = 0

        for rnd in range(self.rounds):
            if rnd == 0 and round0 is not None:          # shared first-round answers (trials)
                outs = [round0[a.agent_id] for a in self.agents]
            elif self.parallel and len(self.agents) > 1:
                with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
                    outs = list(ex.map(lambda a: a.propose(problem, transcript, view), self.agents))
            else:
                outs = [a.propose(problem, transcript, view) for a in self.agents]
            if rnd > 0 and self.carry:
                # Later-round confidence = share of the agent's own independent round-1
                # samples that reached the answer it now gives. An agent that merely
                # adopted someone else's answer gets low confidence for it.
                outs = [t.model_copy(update={"confidence": min(0.99, max(0.01,
                        sum(answers_match(x, t.answer) for x in self.carry[a.agent_id]) / len(self.carry[a.agent_id])))})
                        for a, t in zip(self.agents, outs)]
            turns = [(a.agent_id, t) for a, t in zip(self.agents, outs)]
            for aid, t in turns:
                subs[aid].append(t)
            if self.stop_unanimous and all(answers_match(t.answer, turns[0][1].answer) for _, t in turns):
                # pool_dual: still debate in round 1 if any SAMPLE disagreed with the shared answer
                if not (rnd == 0 and self.policy == "pool_dual" and self.pool and any(
                        not answers_match(x["answer"], turns[0][1].answer)
                        for ss in self.pool.values() for x in ss)):
                    break                                    # everyone agrees: nothing left to debate
            if rnd == self.rounds - 1:
                break                                    # a last-round speaker would influence nothing
            self._spoken = [x["answer"] for x in transcript]
            picks = (self._pick_many(turns, rnd) if self.policy in MULTI
                     else [self._pick(turns, rnd)])
            for spk_id, spk in picks:
                transcript.append({"round": rnd, "speaker": spk_id, "answer": spk.answer,
                                   "confidence": spk.confidence, "rationale": spk.rationale,
                                   "speaker_prestige": view[spk_id]})

        final = self._vote([(a.agent_id, subs[a.agent_id][-1]) for a in self.agents])
        res = DebateResult(final, answers_match(final, problem["answer"]), transcript, dict(subs), before)

        if update_prestige:
            for aid, ts in subs.items():
                for t in ts:
                    self.ledger.update(aid, t.confidence, answers_match(t.answer, problem["answer"]))
        res.prestige_after = self.ledger.snapshot()
        return res

    def _vote(self, last) -> str:
        """Weighted vote, grouping answers by answers_match (not plain string
        equality) so equivalent forms like 14/3 and \\frac{14}{3} combine
        their weight instead of splitting it."""
        groups = []   # list of [representative_answer, total_weight, head_count]
        for aid, t in last:
            w = self.weight(aid, t)
            for g in groups:
                if answers_match(t.answer, g[0]):
                    g[1] += w
                    g[2] += 1
                    break
            else:
                groups.append([t.answer, w, 1])
        if self.policy in MAJORITY:
            # One agent, one vote: an agent persuaded during the debate counts fully (its
            # carried confidence for the new answer is near 0). Prestige x confidence only
            # breaks ties between equally large groups.
            return max(groups, key=lambda g: (g[2], g[1]))[0]
        return max(groups, key=lambda g: g[1])[0]


# ===========================================================================
# 5. EVALUATION — accuracy + stability under three perturbations
# ===========================================================================

@dataclass
class Metrics:
    policy: str
    n_test: int
    accuracy: float
    flip_rate: float
    flip_c2w: float
    flip_w2c: float
    flip_by_kind: dict
    prestige_final: dict
    mean_score: dict
    ece: dict

    def row(self) -> str:
        return (f"{self.policy:16s} acc={self.accuracy:.2f} flip={self.flip_rate:.2f} "
                f"C->W={self.flip_c2w:.2f} W->C={self.flip_w2c:.2f} | "
                f"order={self.flip_by_kind['order']:.2f} drop={self.flip_by_kind['drop']:.2f} "
                f"fresh={self.flip_by_kind['fresh']:.2f}")


def evaluate(make_agents, problems, policy, rounds=3, warmup_frac=0.4, alpha=0.3,
             rule="brier", stake=False, tau=None, seed=0, verbose=False,
             perturbations=("order", "drop", "fresh")) -> Metrics:
    rng = random.Random(seed)
    agents = make_agents()
    ids = [a.agent_id for a in agents]
    ledger = PrestigeLedger(ids, rule=rule, alpha=alpha, stake_by_confidence=stake)
    scaler = PlattScaler()
    orch = Orchestrator(agents, ledger, policy, rounds, tau=tau, scaler=scaler, seed=seed)

    n_warm = int(len(problems) * warmup_frac)
    warm, test = problems[:n_warm], problems[n_warm:]

    for p in warm:
        orch.run(p, update_prestige=True)
    if policy == "posthoc":
        scaler.fit([(c, y) for recs in ledger.records.values() for c, y, _ in recs])
    if policy == "fixed":
        ledger.frozen = True

    n_correct = flips = c2w = w2c = 0
    kind_flips = {"order": 0, "drop": 0, "fresh": 0}
    kind_n = {"order": 0, "drop": 0, "fresh": 0}

    for i, p in enumerate(test):
        canon = orch.run(p, update_prestige=False)
        n_correct += canon.correct

        # Build the list of (kind, orchestrator) jobs to run — independent of
        # each other (all update_prestige=False), so they run concurrently
        # instead of one-after-another. The shared rate limiter still paces
        # the underlying API calls, so this doesn't cause bursting.
        jobs = []
        if "order" in perturbations:
            shuffled = agents[:]
            rng.shuffle(shuffled)
            jobs.append(("order", Orchestrator(shuffled, ledger, policy, rounds, tau, scaler, seed=seed)))
        if "drop" in perturbations:
            for j in range(len(agents)):
                sub = agents[:j] + agents[j + 1:]
                jobs.append(("drop", Orchestrator(sub, ledger, policy, rounds, tau, scaler, seed=seed)))
        if "fresh" in perturbations:
            j = rng.randrange(len(agents))
            fresh = copy.copy(agents[j])
            fresh.agent_id = f"{agents[j].agent_id}_fresh"
            fresh_ledger = copy.deepcopy(ledger)
            fresh_ledger.values[fresh.agent_id] = ledger.prior
            sub = agents[:j] + [fresh] + agents[j + 1:]
            jobs.append(("fresh", Orchestrator(sub, fresh_ledger, policy, rounds, tau, scaler, seed=seed)))

        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                results = list(ex.map(lambda kj: (kj[0], kj[1].run(p, False)), jobs))
        else:
            results = []

        for kind, r in results:
            kind_n[kind] += 1
            if not answers_match(r.final_answer, canon.final_answer):
                flips += 1
                kind_flips[kind] += 1
                if canon.correct and not r.correct:
                    c2w += 1
                elif (not canon.correct) and r.correct:
                    w2c += 1

        orch.run(p, update_prestige=True)      # learn from this problem
        if verbose:
            print(f"  [{policy}] {i+1}/{len(test)} canon={'OK ' if canon.correct else 'BAD'} "
                  f"prestige={ {a: round(v,2) for a,v in ledger.snapshot().items()} }", flush=True)

    n_pert = sum(kind_n.values())
    return Metrics(policy, len(test), n_correct / max(1, len(test)),
                   flips / max(1, n_pert), c2w / max(1, n_pert), w2c / max(1, n_pert),
                   {k: kind_flips[k] / max(1, kind_n[k]) for k in kind_n},
                   ledger.snapshot(),
                   {a: ledger.mean_score(a) for a in ids},
                   {a: ledger.ece(a) for a in ids})


# ===========================================================================
# 6. PROBLEMS & POPULATIONS
# ===========================================================================

def synthetic(n, seed):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = r.randint(12, 99), r.randint(12, 99)
        out.append({"question": f"Compute {a} × {b}. Give only the integer.",
                    "answer": str(a * b),
                    "distractors": [str(a * b + r.choice([-10, 10, a, -b])), str(a + b)]})
    return out


def load_jsonl(path, n=None, levels=None):
    """Accepts {"question","answer"} lines, or MATH-style {"problem","solution","level"}
    lines from which the \\boxed{} answer is extracted.

    levels: optional set/list of ints, e.g. {4,5}, to keep only those MATH
    difficulty levels. Ignored for {"question",...} lines (no level field)."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "question" in d:
                out.append(d)
            else:
                if levels is not None:
                    lvl = d.get("level", "")
                    lvl_num = re.sub(r"\D", "", str(lvl))  # "Level 4" -> "4"
                    if not lvl_num or int(lvl_num) not in levels:
                        continue
                m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", d.get("solution", ""))
                if m:
                    out.append({"question": d["problem"] + "\n\nGive only the final answer.",
                                "answer": m[-1], "level": d.get("level")})
    if levels is not None and not out:
        print(f"WARNING: no problems matched levels={sorted(levels)}. "
              f"Check that {path} has a \"level\" field (e.g. \"Level 4\").", file=sys.stderr)
    random.Random(0).shuffle(out)
    return out[:n] if n else out


def mock_population(seed):
    def make():
        r = random.Random(seed)
        mk = lambda *a, **k: MockAgent(*a, rng=random.Random(r.random()), **k)
        return [mk("bluffer", 0.55, conf_bias=+0.35, conformity=0.2),
                mk("hedger", 0.85, conf_bias=-0.25, conformity=0.3),
                mk("mid_a", 0.65, conf_bias=0.0, conformity=0.5),
                mk("mid_b", 0.65, conf_bias=+0.1, conformity=0.6),
                mk("follower", 0.50, conf_bias=0.0, conformity=0.9)]
    return make


def parse_panel(spec):
    """'careful@gemini-3.8-flash:low,fast@gemini-3.1-flash-lite:none,...' ->
    [(agent_id, persona, model, thinking)]. A repeated persona gets a numbered id."""
    out, seen = [], defaultdict(int)
    for item in spec.split(","):
        persona, rest = item.strip().split("@")
        model, _, thinking = rest.partition(":")
        if persona not in PERSONAS:
            sys.exit(f"--panel: unknown persona {persona!r}; choose from {list(PERSONAS)}")
        seen[persona] += 1
        aid = persona if seen[persona] == 1 else f"{persona}{seen[persona]}"
        out.append((aid, persona, model, thinking or "low"))
    return out


def _agent_kwargs(a):
    if getattr(a, "backend", "gemini") != "openai":
        return GeminiAgent, {"api": a.api}
    extra = json.loads(a.openai_extra) if getattr(a, "openai_extra", None) else None
    base, key = a.base_url, None
    if getattr(a, "provider", None):
        url, env = PROVIDERS[a.provider]
        base = base or url
        if env:
            key = os.environ.get(env)
            if not key:
                sys.exit(f"--provider {a.provider}: set your key first, e.g. export {env}='...'")
        else:
            key = "ollama" if a.provider == "ollama" else "EMPTY"
    return OpenAIAgent, {"base_url": base, "max_tokens": a.max_tokens, "api_key": key,
                         "extra_body": extra, "schema_mode": a.schema_mode,
                         "max_retries": a.max_retries}


def gemini_population(a):
    tl = None if a.thinking == "none" else a.thinking
    Agent, kw = _agent_kwargs(a)
    if Agent is OpenAIAgent:
        tl = None
    if getattr(a, "panel", None):
        spec = parse_panel(a.panel)
        def make_panel():
            return [Agent(aid, persona, model=model, temperature=a.temperature,
                          thinking_level=None if (th == "none" or Agent is OpenAIAgent) else th,
                          log_path=a.log, **kw)
                    for aid, persona, model, th in spec]
        return make_panel
    def make():
        return [Agent(n, n, model=a.model, temperature=a.temperature,
                      thinking_level=tl, log_path=a.log, **kw)
                for n in PERSONAS]
    return make


# ===========================================================================
# 7. COMMANDS
# ===========================================================================

def cmd_smoke(a):
    """One call per transport, full traceback. Use this to debug connectivity."""
    import traceback
    p = {"question": "Compute 17 × 23. Give only the integer.", "answer": "391"}
    if getattr(a, "backend", "gemini") == "openai":
        Agent, kw = _agent_kwargs(a)
        print(f"\n--- backend=openai  base_url={kw['base_url'] or os.environ.get('OPENAI_BASE_URL', 'http://localhost:8000/v1')}"
              f"  model={a.model}")
        try:
            ag = Agent("careful", "careful", model=a.model, temperature=a.temperature,
                       log_path=None, **{**kw, "max_retries": 1})
            t = ag.propose(p, [], {"careful": 0.5})
            ok = "CORRECT" if norm(t.answer) == "391" else "WRONG"
            print(f"SUCCESS [{ok}] answer={t.answer} confidence={t.confidence:.2f}")
            print(f"        rationale: {t.rationale[:100]}")
        except Exception:
            traceback.print_exc()
        return
    for api in ([a.api] if a.api else ["legacy", "interactions"]):
        print(f"\n--- transport={api}  model={a.model or os.environ.get('MAD_MODEL', 'gemini-3.8-flash')}")
        try:
            ag = GeminiAgent("careful", "careful", model=a.model, temperature=a.temperature,
                             thinking_level=(None if a.thinking == "none" else a.thinking),
                             log_path=None, api=api, max_retries=1)
            t = ag.propose(p, [], {"careful": 0.5})
            ok = "CORRECT" if norm(t.answer) == "391" else "WRONG"
            print(f"SUCCESS [{ok}] answer={t.answer} confidence={t.confidence:.2f}")
            print(f"        rationale: {t.rationale[:100]}")
        except Exception:
            traceback.print_exc()
    print("\nIf legacy works and interactions doesn't, just use --api legacy (the default).")


def cmd_mock(a):
    print(f"{'policy':16s}  acc  flip  C->W  W->C | order  drop fresh   (mean of {a.seeds} seeds)")
    for pol in a.policies.split(","):
        ms = [evaluate(mock_population(s), synthetic(a.n, s), pol, a.rounds, alpha=a.alpha,
                       rule=a.rule, stake=a.stake, tau=a.tau, seed=s) for s in range(a.seeds)]
        avg = lambda f: sum(f(m) for m in ms) / len(ms)
        print(f"{pol:16s} {avg(lambda m: m.accuracy):.2f}  {avg(lambda m: m.flip_rate):.2f}  "
              f"{avg(lambda m: m.flip_c2w):.2f}  {avg(lambda m: m.flip_w2c):.2f} | "
              f"{avg(lambda m: m.flip_by_kind['order']):.2f}  {avg(lambda m: m.flip_by_kind['drop']):.2f}  "
              f"{avg(lambda m: m.flip_by_kind['fresh']):.2f}")
        print("   prestige:", {k: round(v, 2) for k, v in ms[-1].prestige_final.items()})


def _levels(a):
    return {int(x) for x in a.level.split(",")} if a.level else None


def cmd_calib(a):
    """MILESTONE 1: single agent, no debate. Does stated confidence predict correctness?"""
    probs = load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0)
    agents = gemini_population(a)()
    led = PrestigeLedger([x.agent_id for x in agents])
    lvl_str = f"levels={sorted(_levels(a))}" if a.level else "levels=all"
    print(f"Running {len(probs)} problems x {len(agents)} agents "
          f"({len(probs)*len(agents)} calls, {a.api or 'legacy'} transport, "
          f"thinking={a.thinking}, {lvl_str})\n")
    for i, p in enumerate(probs):
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            turns = list(ex.map(lambda ag: ag.propose(p, [], led.snapshot()), agents))
        marks = []
        for ag, t in zip(agents, turns):
            ok = answers_match(t.answer, p["answer"])
            led.update(ag.agent_id, t.confidence, ok)
            marks.append(f"{ag.agent_id[:4]}{'+' if ok else '-'}{t.confidence:.2f}")
        print(f"[{i+1}/{len(probs)}] truth={p['answer']:<8} " + "  ".join(marks), flush=True)

    print(f"\n{'agent':12s} {'acc':>5s} {'meanconf':>9s} {'brier':>6s} {'ECE':>5s}")
    for ag in agents:
        r = led.records[ag.agent_id]
        acc = sum(y for _, y, _ in r) / len(r)
        mc = sum(c for c, _, _ in r) / len(r)
        print(f"{ag.agent_id:12s} {acc:5.2f} {mc:9.2f} {led.mean_score(ag.agent_id):6.2f} "
              f"{led.ece(ag.agent_id):5.2f}")
    print("\nRead this: if acc is ~1.00 everywhere, the problems are too easy to show\n"
          "anything — use harder ones. If acc is far below meanconf everywhere (high ECE),\n"
          "confidence isn't carrying signal; fix elicitation before running debates.")


def cmd_debate(a):
    probs = load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0)
    ms = []
    for pol in a.policies.split(","):
        print(f"\n=== {pol}")
        try:
            perts = tuple(a.perturbations.split(","))
            m = evaluate(gemini_population(a), probs, pol, a.rounds, alpha=a.alpha,
                         rule=a.rule, stake=a.stake, tau=a.tau, verbose=True,
                         perturbations=perts, warmup_frac=a.warmup_frac)
            print(m.row())
            ms.append(m)
        except Exception as e:
            print(f"!!! {pol} FAILED: {type(e).__name__}: {e}", flush=True)
            print(f"    Saving {len(ms)} completed polic{'y' if len(ms)==1 else 'ies'} and stopping.")
            break
        finally:
            if ms:
                with open(a.out, "w") as f:
                    json.dump([asdict(x) for x in ms], f, indent=2)
    print("\nwrote", a.out, f"({len(ms)}/{len(a.policies.split(','))} policies completed)")


# ===========================================================================
# 8. TRIALS — multi-seed, no warmup, parallel across problems
# ===========================================================================
#
# `debate` learns prestige problem-by-problem, so problems must run one after
# another. `trials` instead computes prestige from each agent's independent
# FIRST-ROUND answer (given before it has seen anyone else), which makes every
# debate independent of every other:
#   phase A  first-round answers for all problems         (parallel)
#   phase B  prestige-before-problem-i, computed offline from problems < i
#   phase C  rounds 2..N of every (policy, problem) debate  (parallel)
# No warmup: problem i is always scored with prestige learned from problems
# before it, never from itself. All policies share the same first-round
# answers, so the comparison is paired. Everything is appended to a JSONL store
# (resumable, regradable) and summarized by `report`.

TRIAL_POLICIES = ("prestige_x_conf", "conf_only", "prestige_only", "round_robin", "both_sides", "dual", "rr_dual", "pool_dual",
                  "seq_dual", "diss_only", "lead_rand", "dual_randdiss")


def _qid(problem):
    import hashlib
    return hashlib.md5(problem["question"].encode()).hexdigest()[:10]


class _Store:
    """Append-only JSONL: a crash or Ctrl-C loses nothing, and reruns resume."""
    def __init__(self, path):
        self.path, self.lock, self.rows = path, threading.Lock(), []
        if os.path.exists(path):
            with open(path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]

    def add(self, row):
        with self.lock:
            self.rows.append(row)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")


def prestige_path(first_round, ids, mode="shrunk", alpha=0.3, prior=0.5, k=5, rule="brier"):
    """first_round: per problem, in order, {agent: (confidence, correct)}.
    Returns the prestige of every agent BEFORE each problem (learned only from
    earlier problems). mode 'shrunk' = running mean pulled toward the prior with
    weight k; 'ema' = exponential average, one update per problem."""
    # 'contested' = 'shrunk', but only problems where the agents' first answers
    # split (some right, some wrong) count; unanimous problems teach nothing about
    # who to trust.
    score = RULES[rule]
    v = {a: prior for a in ids}
    tot = {a: 0.0 for a in ids}
    path, n = [], 0
    for r in first_round:
        path.append(dict(v))
        right = sum(bool(r[a][1]) for a in ids)
        if mode == "contested" and not 0 < right < len(ids):
            continue
        n += 1
        for a in ids:
            s = score(*r[a])
            if mode == "ema":
                v[a] = (1 - alpha) * v[a] + alpha * s
            else:
                tot[a] += s
                v[a] = (tot[a] + k * prior) / (n + k)
    return path


def _trial_problems(a, seed):
    if a.offset:
        pool = load_jsonl(a.problems, None, levels=_levels(a)) if a.problems else synthetic(a.offset + a.n, 0)
        chunk = pool[a.offset + seed * a.n:a.offset + (seed + 1) * a.n] if a.disjoint else pool[a.offset:a.offset + a.n]
        if len(chunk) < a.n:
            sys.exit(f"--offset {a.offset} --n {a.n} needs more problems than {a.problems} has ({len(pool)}).")
        if seed and not a.disjoint:
            chunk = chunk[:]
            random.Random(seed).shuffle(chunk)
        return chunk
    if a.disjoint:
        pool = load_jsonl(a.problems, None, levels=_levels(a)) if a.problems else synthetic(a.n * (seed + 1), 0)
        chunk = pool[seed * a.n:(seed + 1) * a.n]
        if len(chunk) < a.n:
            sys.exit(f"--disjoint needs {a.n * (seed + 1)} problems; {a.problems} only has {len(pool)}.")
        return chunk
    base = load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0)
    if seed:                                   # seed 0 keeps the standard order
        base = base[:]
        random.Random(seed).shuffle(base)
    return base


def self_consistent(samples):
    """samples: k AgentTurns from one agent on one problem, answered independently.
    Returns the agent's modal answer, with confidence = share of samples that
    agree with it (3/3 -> 0.99, 2/3 -> 0.67, 1/3 -> 0.33). A model that is unsure
    tends to give different answers each time, so agreement carries the signal
    that stated confidence (almost always 0.99) does not."""
    groups = []
    for t in samples:
        for g in groups:
            if answers_match(t.answer, g[0].answer):
                g.append(t); break
        else:
            groups.append([t])
    best = max(groups, key=len)
    rec = best[0].model_dump()
    rec["stated_confidence"] = round(sum(t.confidence for t in best) / len(best), 4)
    rec["confidence"] = min(0.99, max(0.01, len(best) / len(samples)))
    rec["samples"] = [t.answer for t in samples]
    rec["sample_turns"] = [{"answer": t.answer, "rationale": t.rationale,
                            "confidence": round(t.confidence, 4)} for t in samples]
    return rec


def auroc(pos, neg):
    """P(a random right answer has higher confidence than a random wrong one); ties count half.
    0.5 = confidence says nothing about correctness, 1.0 = perfectly separates them."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def _run_pool(workers, jobs, label):
    """Run zero-arg callables concurrently; a failed task is reported, not fatal."""
    from concurrent.futures import as_completed
    if not jobs:
        return 0
    failed, t0 = 0, time.time()
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print(f"  !! {label} task failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                print(f"  [{label}] {k}/{len(jobs)} done, {failed} failed, {time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: stopped. Debates not yet saved will be redone. "
              "Rerun the same command to resume.", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    ex.shutdown(wait=True)
    return failed


def cmd_trials(a):
    pols = a.policies.split(",")
    if a.policies == ",".join(POLICIES):
        pols = ["round_robin", "prestige_x_conf"]          # default: the main comparison
    bad = [p for p in pols if p not in TRIAL_POLICIES]
    if bad:
        sys.exit(f"trials supports {TRIAL_POLICIES}; {bad} need a warmup phase (use `debate`).")
    if not a.mock and not a.model and not a.panel:
        sys.exit("trials: pass --model (e.g. gemini-3.1-flash-lite, or Qwen/Qwen2.5-7B-Instruct with --backend openai) or --panel.")
    if a.init_from and not os.path.exists(a.store):
        # start a new store from another run's round-1 answers (same problems, no new round-1 calls)
        with open(a.init_from) as src, open(a.store, "w") as dst:
            for line in src:
                if line.strip() and json.loads(line)["kind"] in ("config", "r0"):
                    dst.write(line)
        print(f"initialized {a.store} with round-1 answers from {a.init_from}")
    store = _Store(a.store)
    cfg = {"model": "mock" if a.mock else a.model, "problems": a.problems, "n": a.n, "level": a.level,
           "rounds": a.rounds, "temperature": a.temperature, "thinking": a.thinking,
           "prestige_mode": a.prestige_mode, "alpha": a.alpha, "prior_strength": a.prior_strength,
           "rule": a.rule, "tau": a.tau, "disjoint": a.disjoint}
    for k, v in (("panel", a.panel), ("offset", a.offset), ("stop_unanimous", a.stop_unanimous),
                 ("sc_k", a.sc_k if a.sc_k > 1 else None), ("sc_carry", a.sc_carry),
                 ("conf_mix", a.conf_mix if a.conf_mix != "sc" else None),
                 ("warmup_from", os.path.basename(a.warmup_from) if a.warmup_from else None),
                 ("warmup_n", a.warmup_n),
                 ("backend", a.backend if a.backend != "gemini" else None),
                 ("provider", a.provider)):
        if v:                                    # only recorded when used, so old stores still resume
            cfg[k] = v
    old = [r for r in store.rows if r["kind"] == "config"]
    # A larger --n just adds problems. Before any debate has run, settings that only
    # affect debates (rounds, prestige rule, ...) may still change: round-1 answers don't depend on them.
    round1_keys = {"model", "problems", "level", "temperature", "thinking", "disjoint", "panel", "offset", "sc_k",
                   "backend", "provider"}
    debated = any(r["kind"] == "debate" for r in store.rows)
    same = lambda c: {k: v for k, v in c.items() if k != "n" and (debated or k in round1_keys)}
    if old and same(old[-1]["cfg"]) != same(cfg):
        diff = {k: (old[-1]["cfg"].get(k), v) for k, v in cfg.items() if k != "n" and old[-1]["cfg"].get(k) != v}
        sys.exit(f"{a.store} was started with different settings {diff} (stored, now).\n"
                 f"Use a new --store file, or rerun with the original settings to resume.")
    if not old:
        store.add({"kind": "config", "cfg": cfg, "started": time.time()})
    elif not any(r["kind"] == "debate" for r in store.rows) and old[-1]["cfg"] != cfg:
        old[-1]["cfg"] = cfg                       # debate settings changed before any debate: record the new ones
        store.add({"kind": "config", "cfg": cfg, "started": time.time(), "note": "debate settings updated"})

    n_ag = len(parse_panel(a.panel)) if a.panel else 5
    later = max(0, a.rounds - 1)                 # rounds after the shared first one make calls
    per_debate = 2 * later * n_ag                # canonical + reordered run
    r0_rows = [r for r in store.rows if r["kind"] == "r0"]
    n_r0 = len(r0_rows)
    split = sum(1 for r in r0_rows if len({norm(t["answer"]) for t in r["turns"].values()}) > 1)
    est_split = split / n_r0 if (n_r0 and a.stop_unanimous) else (0.5 if a.stop_unanimous else 1.0)
    total_r0 = a.seeds * a.n * n_ag * a.sc_k
    total_deb = int(a.seeds * a.n * est_split * len(pols) * per_debate) if not a.round1_only else 0
    have_deb = sum(r.get("calls", per_debate) for r in store.rows if r["kind"] == "debate")
    left = max(0, total_r0 - n_r0 * n_ag * a.sc_k) + max(0, total_deb - have_deb)
    print(f"{'panel ' + a.panel if a.panel else 'model ' + cfg['model']}")
    print(f"{a.seeds} seed(s) x {a.n} problems (offset {a.offset}) | policies {pols} | up to {a.rounds} rounds | "
          f"prestige={a.prestige_mode} | stop when unanimous: {a.stop_unanimous} | workers {a.workers} | "
          f"round-1 confidence: {'self-consistency, %d samples' % a.sc_k if a.sc_k > 1 else 'stated'}")
    if a.stop_unanimous:
        print(f"split first answers: {'%d/%d measured' % (split, n_r0) if n_r0 else 'unknown yet, assuming 50%'}")
    print(f"calls: about {left} still to make (upper bound; ~${left * a.price_per_call:.2f} at "
          f"${a.price_per_call}/call -- stronger models cost more per call; "
          f"at least {left * a.stagger / 60:.0f} min at this stagger)")
    print(f"results -> {a.store} (append-only; rerun the same command to resume)\n")
    if a.dry_run:
        return

    warm_rows, lab = [], (lambda pol: pol)
    if a.warmup_from:
        wr = [r for r in _Store(a.warmup_from).rows if r["kind"] == "r0"]
        if not wr:
            sys.exit(f"{a.warmup_from} has no round-1 answers; make it with `trials --round1-only`.")
        test_q = {_qid(p) for s_ in range(a.seeds) for p in _trial_problems(a, s_)}
        if test_q & {r["qid"] for r in wr}:
            sys.exit("warmup problems overlap the test problems; use a separate problem set.")
        warm_rows = sorted([r for r in wr if r["seed"] == min(x["seed"] for x in wr)], key=lambda r: r["pid"])
        if a.warmup_n:
            if a.warmup_n > len(warm_rows):
                sys.exit(f"--warmup-n {a.warmup_n} but {a.warmup_from} has only {len(warm_rows)} problems.")
            warm_rows = warm_rows[:a.warmup_n]
        tag = f"_warm{len(warm_rows)}"
        lab = lambda pol: pol + tag
        print(f"warmup: prestige starts from {len(warm_rows)} problems in {a.warmup_from}")

    for seed in range(a.seeds):
        probs = _trial_problems(a, seed)
        agents = mock_population(seed)() if a.mock else gemini_population(a)()
        ids = [x.agent_id for x in agents]
        qids = [_qid(p) for p in probs]
        print(f"=== seed {seed}")

        def do_r0(i, probs=probs, agents=agents, qids=qids, seed=seed):
            p, k = probs[i], a.sc_k
            with ThreadPoolExecutor(max_workers=len(agents) * k) as ex:
                outs = list(ex.map(lambda ag: ag.propose(p, [], None), [ag for ag in agents for _ in range(k)]))
            if k == 1:
                turns = {ag.agent_id: t.model_dump() for ag, t in zip(agents, outs)}
            else:
                turns = {ag.agent_id: self_consistent(outs[j * k:(j + 1) * k]) for j, ag in enumerate(agents)}
            store.add({"kind": "r0", "seed": seed, "qid": qids[i], "pid": i, "truth": p["answer"],
                       "level": p.get("level"), "turns": turns})

        for attempt in range(3):                      # automatic retry passes for failed tasks
            done = {r["qid"] for r in store.rows if r["kind"] == "r0" and r["seed"] == seed}
            jobs = [(lambda i=i: do_r0(i)) for i in range(len(probs)) if qids[i] not in done]
            if not _run_pool(a.workers, jobs, f"seed {seed} round 1" + (f" retry {attempt}" if attempt else "")):
                break
        r0 = {r["qid"]: r for r in store.rows if r["kind"] == "r0" and r["seed"] == seed}
        missing = [i for i in range(len(probs)) if qids[i] not in r0]
        if missing:
            sys.exit(f"{len(missing)} problems still lack first-round answers after 3 passes (API trouble). "
                     f"Rerun the same command later to resume; nothing is lost.")
        if a.round1_only:
            continue

        mix = (lambda t: (t["confidence"] + t.get("stated_confidence", t["confidence"])) / 2) \
            if a.conf_mix == "avg" else (lambda t: t["confidence"])
        fr = [{aid: (mix(r0[qids[i]]["turns"][aid]),
                     answers_match(r0[qids[i]]["turns"][aid]["answer"], probs[i]["answer"])) for aid in ids}
              for i in range(len(probs))]
        pre = [{aid: (mix(w["turns"][aid]), answers_match(w["turns"][aid]["answer"], w["truth"])) for aid in ids}
               for w in warm_rows]
        path = prestige_path(pre + fr, ids, a.prestige_mode, a.alpha, k=a.prior_strength, rule=a.rule)[len(pre):]

        if "pool_dual" in pols and not all("sample_turns" in t for r in r0.values() for t in r["turns"].values()):
            sys.exit("pool_dual needs round-1 sample rationales, which older stores don't have.\n"
                     "Start a NEW --store with mad_gemini_v4.py (no --init-from an older store).")

        def do_debate(pol, i, probs=probs, agents=agents, ids=ids, qids=qids, path=path, r0=r0, seed=seed):
            p, qid = probs[i], qids[i]
            led = PrestigeLedger(ids, rule=a.rule, values=dict(path[i]), frozen=True)
            first = {aid: AgentTurn(**{**r0[qid]["turns"][aid], "confidence": mix(r0[qid]["turns"][aid])})
                     for aid in ids}
            pool = ({aid: r0[qid]["turns"][aid]["sample_turns"] for aid in ids}
                    if all("sample_turns" in r0[qid]["turns"][aid] for aid in ids) else None)
            carry = ({aid: r0[qid]["turns"][aid]["samples"] for aid in ids}
                     if a.sc_carry and all("samples" in r0[qid]["turns"][aid] for aid in ids) else None)
            mk = lambda ags, tag: Orchestrator(ags, led, pol, a.rounds, tau=a.tau, random_ties=True,
                                               seed=f"{seed}:{qid}:{tag}", stop_unanimous=a.stop_unanimous,
                                               carry=carry,
                                               pool=pool if pol == "pool_dual" else None)
            perm = agents[:]
            random.Random(f"{seed}:{qid}").shuffle(perm)
            with ThreadPoolExecutor(max_workers=2) as ex:      # canonical + reordered run side by side
                f_canon = ex.submit(mk(agents, "canon").run, p, False, first)
                f_pert = ex.submit(mk(perm, "order").run, p, False, first)
                canon, pert = f_canon.result(), f_pert.result()
            calls = sum(len(v) - 1 for v in canon.submissions.values()) + \
                    sum(len(v) - 1 for v in pert.submissions.values())
            store.add({"kind": "debate", "seed": seed, "policy": lab(pol), "qid": qid, "pid": i,
                       "truth": p["answer"], "prestige": path[i], "calls": calls,
                       "canon": {"final": canon.final_answer,
                                 "speakers": [t["speaker"] for t in canon.transcript],
                                 "speaker_rounds": [t["round"] for t in canon.transcript],
                                 "last": {k: v[-1].answer for k, v in canon.submissions.items()},
                                 # every agent's answer and confidence in every round (per-round metrics)
                                 "rounds": {k: [[t.answer, round(t.confidence, 4)] for t in v]
                                            for k, v in canon.submissions.items()}},
                       "order": {"final": pert.final_answer, "perm": [x.agent_id for x in perm],
                                 "speakers": [t["speaker"] for t in pert.transcript]}})

        for attempt in range(3):
            done = {(r["policy"], r["qid"]) for r in store.rows if r["kind"] == "debate" and r["seed"] == seed}
            jobs = [(lambda pol=pol, i=i: do_debate(pol, i)) for i in range(len(probs)) for pol in pols
                    if (lab(pol), qids[i]) not in done]
            if not _run_pool(a.workers, jobs, f"seed {seed} debates" + (f" retry {attempt}" if attempt else "")):
                break
        else:
            print(f"WARNING: seed {seed} still has failed debates after 3 passes; "
                  f"rerun the same command later to fill them in.", flush=True)

    if a.round1_only:
        report_round1(store.rows)
        print("\nRound 1 done. Rerun the same command without --round1-only to run the debates "
              "(first answers are reused, not repaid).")
        return
    report_trials(store.rows, pols, out=a.out)


def report_round1(rows):
    """Per-agent accuracy on independent first answers: does the panel really differ?"""
    cfg = next((r["cfg"] for r in rows if r["kind"] == "config"), {})
    models = {aid: f"{m}:{th}" for aid, _, m, th in parse_panel(cfg["panel"])} if cfg.get("panel") else {}
    r0 = [r for r in rows if r["kind"] == "r0"]
    if not r0:
        return
    ids = list(r0[0]["turns"])
    print(f"\nround-1 accuracy per agent ({len(r0)} problem-runs)")
    for aid in ids:
        ok = [answers_match(r["turns"][aid]["answer"], r["truth"]) for r in r0]
        conf = [r["turns"][aid]["confidence"] for r in r0]
        print(f"  {aid:12s} {models.get(aid, ''):34s} acc {sum(ok) / len(ok):.3f}   mean conf {sum(conf) / len(conf):.2f}")
    # Calibration check: does confidence separate right answers from wrong ones?
    pts = [(t, answers_match(t["answer"], r["truth"])) for r in r0 for t in r["turns"].values()]
    acc_all = sum(ok for _, ok in pts) / len(pts)
    print(f"\nconfidence check over {len(pts)} answers (overall accuracy {acc_all:.3f})")
    measures = [("stated confidence", lambda t: round(t.get("stated_confidence", t["confidence"]), 3))]
    if any("samples" in t for t, _ in pts):
        measures.append(("self-consistency", lambda t: t["confidence"]))
    for name, f in measures:
        right = [f(t) for t, ok in pts if ok]
        wrong = [f(t) for t, ok in pts if not ok]
        print(f"  {name:18s} AUROC {auroc(right, wrong):.3f}   (0.5 = no signal; 0.75+ = useful)")
        levels = defaultdict(list)
        for t, ok in pts:
            levels[round(f(t), 2)].append(ok)
        for c in sorted(levels):
            if len(levels[c]) >= 5 or name == "self-consistency":
                print(f"      conf {c:.2f}: {len(levels[c]):5d} answers, {sum(levels[c]) / len(levels[c]):.0%} right")
    k = [sum(answers_match(t["answer"], r["truth"]) for t in r["turns"].values()) for r in r0]
    split = sum(len({norm(t["answer"]) for t in r["turns"].values()}) > 1 for r in r0)
    print(f"  contested (some right, some wrong): {sum(0 < x < len(ids) for x in k)}/{len(r0)}   "
          f"all wrong: {sum(x == 0 for x in k)}   first answers not unanimous: {split}/{len(r0)}")


def report_trials(rows, pols=None, ref="round_robin", B=2000, out=None):
    """Summaries and paired comparisons from a trials store. Correctness is
    recomputed from stored answers with the CURRENT grader, so fixing the grader
    never requires rerunning the model."""
    import statistics as st
    r0 = {(r["seed"], r["qid"]): r for r in rows if r["kind"] == "r0"}
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate":
            deb[r["policy"]][(r["seed"], r["qid"])] = r
    pols = [p for p in (pols or sorted(deb)) if p in deb]
    if not pols:
        print("No completed debates in the store yet."); return
    seeds = sorted({s for s, _ in r0})
    npid = max(r["pid"] for r in r0.values()) + 1

    def derive(rec):
        t = rec["truth"]
        acc = answers_match(rec["canon"]["final"], t)
        pok = answers_match(rec["order"]["final"], t)
        flip = not answers_match(rec["canon"]["final"], rec["order"]["final"])
        turns = r0[(rec["seed"], rec["qid"])]["turns"]
        k = sum(answers_match(v["answer"], t) for v in turns.values())
        return dict(acc=float(acc), flip=float(flip), c2w=float(flip and acc and not pok),
                    w2c=float(flip and (not acc) and pok), contested=0 < k < len(turns),
                    late=rec["pid"] >= npid // 2)

    D = {p: {key: derive(rec) for key, rec in recs.items()} for p, recs in deb.items() if p in pols}
    per_seed_n = {sd: sum(1 for s, _ in r0 if s == sd) for sd in seeds}
    print("completeness (debates done / problems with round-1 answers):")
    for sd in seeds:
        print(f"  seed {sd}: " + ", ".join(f"{p} {sum(1 for s, _ in D[p] if s == sd)}/{per_seed_n[sd]}" for p in pols))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    def by_seed(p, field, sel=lambda d: True):
        return [mean([d[field] for (s, _), d in D[p].items() if s == sd and sel(d)]) for sd in seeds]

    def fmt(xs):
        xs = [x for x in xs if x == x]
        if not xs: return "   n/a"
        sd = f" +-{st.stdev(xs):.3f}" if len(xs) > 1 else ""
        return f"{mean(xs):.3f}{sd}"

    report_round1(rows)
    print(f"\n{len(seeds)} seed(s); mean +- sd across seeds; every problem scored (no warmup)\n")
    print(f"{'':30s}{'accuracy':>16s}{'flip':>16s}{'C->W':>14s}{'W->C':>14s}{'acc 2nd half':>16s}{'acc contested':>16s}")
    base_single, base_maj = [], []
    for sd in seeds:
        single, maj = [], []
        for (s, qid), r in r0.items():
            if s != sd: continue
            t = r["truth"]
            oks = [answers_match(v["answer"], t) for v in r["turns"].values()]
            single.append(mean([float(o) for o in oks]))
            groups = []
            for v in r["turns"].values():
                for g in groups:
                    if answers_match(v["answer"], g[0]):
                        g[1] += 1; break
                else:
                    groups.append([v["answer"], 1])
            maj.append(float(answers_match(max(groups, key=lambda g: g[1])[0], t)))
        base_single.append(mean(single)); base_maj.append(mean(maj))
    print(f"{'single agent (round 1)':30s}{fmt(base_single):>16s}")
    print(f"{'majority vote, no debate':30s}{fmt(base_maj):>16s}")
    # Weighted votes over the round-1 answers alone (no debate, no extra calls).
    prest = {}
    for recs in deb.values():
        for key, rec in recs.items():
            prest.setdefault(key, rec["prestige"])
    def wvote(r, w):
        groups = []
        for aid, t in r["turns"].items():
            for g in groups:
                if answers_match(t["answer"], g[0]):
                    g[1] += w(aid, t); break
            else:
                groups.append([t["answer"], w(aid, t)])
        return float(answers_match(max(groups, key=lambda g: g[1])[0], r["truth"]))
    nd = {"confidence vote, no debate": lambda key: (lambda aid, t: t["confidence"]),
          "prestige*conf vote, no debate": lambda key: (lambda aid, t: prest[key].get(aid, 0.5) * t["confidence"])}
    for label, mk in nd.items():
        vals = [mean([wvote(r, mk(key)) for key, r in r0.items() if key[0] == sd and key in prest]) for sd in seeds]
        print(f"{label:30s}{fmt(vals):>16s}")
    summary = {"seeds": seeds, "baseline_single": base_single, "baseline_majority": base_maj, "policies": {}}
    for p in pols:
        cols = [by_seed(p, "acc"), by_seed(p, "flip"), by_seed(p, "c2w"), by_seed(p, "w2c"),
                by_seed(p, "acc", lambda d: d["late"]), by_seed(p, "acc", lambda d: d["contested"])]
        print(f"{p:30s}" + "".join(f"{fmt(c):>{w}s}" for c, w in zip(cols, [16, 16, 14, 14, 16, 16])))
        summary["policies"][p] = dict(zip(["acc", "flip", "c2w", "w2c", "acc_late", "acc_contested"], cols))

    # Free check: round robin's own final answers, re-voted with prestige weights.
    if "round_robin" in deb:
        rv = defaultdict(list)
        for (s, q), rec in deb["round_robin"].items():
            last = rec["canon"].get("last")
            if not last:
                continue
            groups = []
            for aid, ans in last.items():
                w = rec["prestige"].get(aid, 0.5)
                for g in groups:
                    if answers_match(ans, g[0]):
                        g[1] += w; break
                else:
                    groups.append([ans, w])
            rv[s].append(float(answers_match(max(groups, key=lambda g: g[1])[0], rec["truth"])))
        if rv:
            print(f"{'round_robin + prestige vote':30s}{fmt([mean(rv[s]) for s in seeds if rv[s]]):>16s}"
                  f"   (offline: same debates, final vote weighted by prestige)")

    if ref in pols and len(pols) > 1:
        print(f"\npaired differences vs {ref} (95% bootstrap CI over problems; problems shared across seeds are resampled together)")
        rng = random.Random(0)
        for p in pols:
            if p == ref: continue
            keys = sorted(set(D[p]) & set(D[ref]))
            qs = sorted({q for _, q in keys})
            row = f"  {p:18s}"
            for field in ["acc", "flip", "c2w"]:
                per_q = {q: mean([D[p][(s, q)][field] - D[ref][(s, q)][field] for s in seeds if (s, q) in D[p] and (s, q) in D[ref]])
                         for q in qs}
                vals = list(per_q.values())
                boots = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(B))
                lo, hi = boots[int(.025 * B)], boots[int(.975 * B) - 1]
                row += f"  d{field} {mean(vals):+.3f} [{lo:+.3f}, {hi:+.3f}]"
                summary["policies"][p][f"d_{field}_vs_{ref}"] = [mean(vals), lo, hi]
            print(row)
            # McNemar exact test: only problems where exactly one policy was right carry information.
            b = sum(1 for k in keys if D[p][k]["acc"] and not D[ref][k]["acc"])
            c = sum(1 for k in keys if D[ref][k]["acc"] and not D[p][k]["acc"])
            n_bc = b + c
            pval = min(1.0, 2 * sum(math.comb(n_bc, i) for i in range(min(b, c) + 1)) / 2 ** n_bc) if n_bc else 1.0
            print(f"  {'':18s}  McNemar exact: {p} right & {ref} wrong on {b}, the reverse on {c}  ->  p = {pval:.3f}")
            summary["policies"][p]["mcnemar"] = {"b": b, "c": c, "p": pval}
        print("  (an interval containing 0 means the difference is not distinguishable from noise)")

    # Mechanism: on problems where first answers disagreed, how often did the
    # chosen speaker hold the right answer?
    print("\nspeaker analysis (problems where first answers disagreed and the debate ran)")
    for p in pols:
        hit = []
        for key, rec in deb[p].items():
            spk = rec["canon"]["speakers"]
            if not spk:
                continue
            t = r0[key]["turns"][spk[0]]
            hit.append(answers_match(t["answer"], rec["truth"]))
        if hit:
            print(f"  {p:18s} first speaker was right on {sum(hit)}/{len(hit)} = {sum(hit) / len(hit):.0%}")
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print("\nwrote", out)


def cmd_resample(a):
    """Stability without reordering: rerun every debated problem with the SAME agent
    order and fresh model calls, and count how often the final answer changes.
    Unlike the reorder test, this does not penalize round robin by construction."""
    src = _Store(a.store)
    cfgs = [r["cfg"] for r in src.rows if r["kind"] == "config"]
    if not cfgs:
        sys.exit(f"{a.store} has no config row; is it a trials store?")
    cfg = cfgs[-1]
    for k in ("problems", "level", "rounds", "temperature", "thinking", "tau", "panel"):
        setattr(a, k, cfg.get(k, getattr(a, k, None)))
    a.model = None if cfg.get("model") == "mock" else cfg.get("model")
    carry_on, stop_on = bool(cfg.get("sc_carry")), bool(cfg.get("stop_unanimous"))
    pols = a.policies.split(",") if a.policies != ",".join(POLICIES) else None
    out = _Store(a.resample_store)
    done = {(r["seed"], r["policy"], r["qid"]) for r in out.rows if r["kind"] == "resample"}
    r0 = {(r["seed"], r["qid"]): r for r in src.rows if r["kind"] == "r0"}
    debs = [r for r in src.rows if r["kind"] == "debate" and r["canon"]["speakers"]
            and (pols is None or r["policy"] in pols)]
    pool = load_jsonl(a.problems, None, levels=_levels(a)) if a.problems else synthetic(1000, 0)
    byq = {_qid(p): p for p in pool}
    mock = cfg.get("model") == "mock"
    agents = mock_population(0)() if mock else gemini_population(a)()
    ids = [x.agent_id for x in agents]
    todo = [r for r in debs if (r["seed"], r["policy"], r["qid"]) not in done]
    n_ag = len(ids)
    print(f"resampling {len(debs)} debated problem-runs ({len(todo)} left), same agent order, fresh calls; "
          f"up to {len(todo) * max(0, a.rounds - 1) * n_ag} calls (~${len(todo) * max(0, a.rounds - 1) * n_ag * a.price_per_call:.2f})")
    if a.dry_run:
        return

    def job(rec):
        key = (rec["seed"], rec["qid"])
        p = byq[rec["qid"]]
        led = PrestigeLedger(ids, rule=cfg.get("rule", "brier"), values=dict(rec["prestige"]), frozen=True)
        turns = r0[key]["turns"]
        first = {aid: AgentTurn(**turns[aid]) for aid in ids}
        carry = ({aid: turns[aid]["samples"] for aid in ids}
                 if carry_on and all("samples" in turns[aid] for aid in ids) else None)
        res = Orchestrator(agents, led, rec["policy"], a.rounds, tau=a.tau, random_ties=True,
                           seed=f"{rec['seed']}:{rec['qid']}:resample", stop_unanimous=stop_on,
                           carry=carry).run(p, False, first)
        out.add({"kind": "resample", "seed": rec["seed"], "policy": rec["policy"], "qid": rec["qid"],
                 "truth": rec["truth"], "final": res.final_answer, "canon_final": rec["canon"]["final"],
                 "order_final": rec["order"]["final"]})

    for attempt in range(3):
        done = {(r["seed"], r["policy"], r["qid"]) for r in out.rows if r["kind"] == "resample"}
        jobs = [(lambda r=r: job(r)) for r in debs if (r["seed"], r["policy"], r["qid"]) not in done]
        if not _run_pool(a.workers, jobs, "resample" + (f" retry {attempt}" if attempt else "")):
            break
    report_resample(out.rows)


def report_resample(rows, ref="round_robin", B=4000):
    recs = [r for r in rows if r["kind"] == "resample"]
    by = defaultdict(dict)
    for r in recs:
        ok_c = answers_match(r["canon_final"], r["truth"])
        ok_n = answers_match(r["final"], r["truth"])
        flip = not answers_match(r["final"], r["canon_final"])
        oflip = not answers_match(r["order_final"], r["canon_final"])
        by[r["policy"]][(r["seed"], r["qid"])] = dict(flip=float(flip), c2w=float(flip and ok_c and not ok_n),
                                                      w2c=float(flip and ok_n and not ok_c), oflip=float(oflip))
    print("\nresampling test on debated problem-runs (same order, fresh calls)")
    print(f"{'policy':18s}{'n':>5s}{'resample flip':>15s}{'C->W':>8s}{'W->C':>8s}{'reorder flip':>15s}")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for p, d in sorted(by.items()):
        v = list(d.values())
        print(f"{p:18s}{len(v):5d}{mean([x['flip'] for x in v]):15.1%}{mean([x['c2w'] for x in v]):8.1%}"
              f"{mean([x['w2c'] for x in v]):8.1%}{mean([x['oflip'] for x in v]):15.1%}")
    rng = random.Random(0)
    for p in by:
        for q in by:
            if p == q or q not in (ref, "conf_only") or (q == "conf_only" and p != "prestige_x_conf"):
                continue
            keys = sorted(set(by[p]) & set(by[q]))
            if not keys:
                continue
            per = defaultdict(list)
            for k in keys:
                per[k[1]].append(by[p][k]["flip"] - by[q][k]["flip"])
            vals = [mean(x) for x in per.values()]
            bs = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(B))
            print(f"  resample flip {p} - {q}: {mean(vals) * 100:+.1f} points "
                  f"[{bs[int(.025 * B)] * 100:+.1f}, {bs[int(.975 * B) - 1] * 100:+.1f}] over {len(keys)} problem-runs")
    print("  (reorder flip on the same problem-runs shown for comparison; an interval containing 0 = no clear difference)")


def cmd_report(a):
    class _Merged:                                   # --store a.jsonl,b.jsonl -> one combined report
        rows = [r for path in a.store.split(",") for r in _Store(path).rows]
    store = _Merged()
    report_trials(store.rows, a.policies.split(",") if a.policies != ",".join(POLICIES) else None, out=a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "mock", "calib", "debate", "trials", "report", "resample"])
    ap.add_argument("--problems", default=None, help="JSONL file; omit for synthetic arithmetic")
    ap.add_argument("--level", default=None, help="MATH levels to keep, e.g. '4,5' (comma-separated)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=None, help="repeats: mock (default 5) / trials (default 3)")
    ap.add_argument("--alpha", type=float, default=0.3, help="prestige EMA rate")
    ap.add_argument("--rule", default="brier", choices=["brier", "log"])
    ap.add_argument("--stake", action="store_true", help="scale prestige update by stated confidence")
    ap.add_argument("--tau", type=float, default=None, help="softmax speaker sampling temperature")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--model", default=None, help="default gemini-3.8-flash; env MAD_MODEL")
    ap.add_argument("--api", default=None, choices=["legacy", "interactions"],
                    help="default legacy; env MAD_API")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--thinking", default="low", choices=["low", "high", "none"],
                    help="default low; 'none' disables the flag entirely")
    ap.add_argument("--log", default="calls.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--stagger", type=float, default=0.3,
                    help="seconds between call starts across all agents (paces bursts to the API)")
    ap.add_argument("--perturbations", default="order,drop,fresh",
                    help="which stability checks to run (comma-sep, subset of order,drop,fresh). "
                        "'drop' is 5x cost (one run per agent removed) — drop it for a faster/"
                        "cheaper first pass, e.g. --perturbations order,fresh")
    ap.add_argument("--warmup-frac", type=float, default=0.4, help="debate: share of problems used as warmup")
    ap.add_argument("--workers", type=int, default=6, help="trials: debates run concurrently")
    ap.add_argument("--store", default="trials.jsonl",
                    help="trials/report: append-only results file (report accepts several, comma-separated)")
    ap.add_argument("--prestige-mode", default="shrunk", choices=["shrunk", "ema", "contested"],
                    help="trials: 'shrunk' = running mean pulled toward the prior; 'ema' uses --alpha")
    ap.add_argument("--prior-strength", type=float, default=5.0, help="trials: pseudo-problems of prior in 'shrunk'")
    ap.add_argument("--disjoint", action="store_true", help="trials: a different block of --n problems per seed")
    ap.add_argument("--mock", action="store_true", help="trials: simulated agents, no API calls")
    ap.add_argument("--price-per-call", type=float, default=0.00065, help="trials: $ per call, for the estimate only")
    ap.add_argument("--panel", default=None,
                    help="trials: mixed panel, e.g. 'careful@gemini-3.8-flash:low,fast@gemini-3.1-flash-lite:none'")
    ap.add_argument("--offset", type=int, default=0, help="trials: skip the first N problems (use fresh ones)")
    ap.add_argument("--stop-unanimous", action="store_true", help="trials: end a debate once every agent agrees")
    ap.add_argument("--resample-store", default="resample.jsonl", help="resample: output file")
    ap.add_argument("--conf-mix", default="sc", choices=["sc", "avg"],
                    help="trials: round-1 confidence = self-consistency (sc) or its average with stated confidence (avg)")
    ap.add_argument("--warmup-from", default=None,
                    help="trials: store with round-1 answers on SEPARATE problems (made with --round1-only); "
                         "prestige starts from that track record. Results are labelled <policy>_warm.")
    ap.add_argument("--warmup-n", type=int, default=None,
                    help="trials: use only the first N problems of --warmup-from (label <policy>_warmN)")
    ap.add_argument("--init-from", default=None, help="trials: copy round-1 answers from this store into a new --store")
    ap.add_argument("--sc-k", type=int, default=1,
                    help="trials: round-1 samples per agent; >1 = confidence from self-consistency (agreement)")
    ap.add_argument("--sc-carry", action="store_true",
                    help="trials: in later rounds, confidence = share of the agent's round-1 samples matching its current answer")
    ap.add_argument("--round1-only", action="store_true",
                    help="trials: only collect independent first answers (cheap panel check); rerun without it to debate")
    ap.add_argument("--dry-run", action="store_true", help="trials: print the plan and cost, make no calls")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "openai"],
                    help="gemini (default) or openai = any OpenAI-compatible server, e.g. vLLM on a GPU")
    ap.add_argument("--base-url", default=None, help="openai backend: server URL (default $OPENAI_BASE_URL or http://localhost:8000/v1)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="openai backend: max output tokens per call")
    ap.add_argument("--schema-mode", default="json_schema", choices=["json_schema", "json_object", "guided_json", "none"],
                    help="openai backend: how to request JSON; json_schema falls back to json_object, then plain text")
    ap.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                    help="hosted/local open models: " + ", ".join(sorted(PROVIDERS)) + " (implies --backend openai)")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="openai backend: tries per call (backoff doubles each time; raise to 8 for free tiers)")
    ap.add_argument("--openai-extra", default=None,
                    help='openai backend: extra JSON body, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\' for Qwen3')
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = 3 if a.cmd == "trials" else 5
    if a.provider:
        a.backend = "openai"
    _RATE_LIMITER.min_interval = a.stagger
    {"smoke": cmd_smoke, "mock": cmd_mock, "calib": cmd_calib, "debate": cmd_debate,
     "trials": cmd_trials, "report": cmd_report, "resample": cmd_resample}[a.cmd](a)


if __name__ == "__main__":
    main()