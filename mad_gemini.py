#!/usr/bin/env python3
"""
mad_gemini.py — prestige-weighted, confidence-scored multi-agent debate, one file.

Setup:
    pip install google-genai pydantic
    export GEMINI_API_KEY="..."

Use:
    python3 mad_gemini.py smoke                     # 1 call per transport, full errors
    python3 mad_gemini.py mock                      # mechanism sanity check, no API
    python3 mad_gemini.py calib --n 20              # does stated confidence track correctness?
    python3 mad_gemini.py debate --problems math.jsonl --n 15

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
        standings = ", ".join(f"{a}: {p:.2f}" for a, p in sorted(prestige.items()))
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

POLICIES = ["prestige_x_conf", "conf_only", "posthoc", "prestige_only", "round_robin", "fixed"]


def _strip_latex(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("$", "").replace("\\!", "").replace("^\\circ", "")
    s = s.replace("\\", "")
    s = s.strip().replace(" ", "")
    return s


def norm(ans) -> str:
    return _strip_latex(ans).lower().replace(",", "").rstrip(".")


def answers_match(a, b) -> bool:
    """String match after LaTeX normalization; falls back to sympy for
    algebraic/numeric equivalence (e.g. 14/3 == \\frac{14}{3} == 4.6667)."""
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    try:
        import sympy
        ea, eb = sympy.sympify(na), sympy.sympify(nb)
        if sympy.simplify(ea - eb) == 0:
            return True
    except Exception:
        pass
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
                 tau=None, scaler: PlattScaler = None, parallel=True, seed=0):
        self.agents, self.ledger, self.policy, self.rounds = agents, ledger, policy, rounds
        self.tau = tau                       # None -> argmax speaker; float -> softmax
        self.scaler = scaler or PlattScaler()
        self.parallel = parallel
        self.rng = random.Random(seed)

    def weight(self, agent_id: str, turn: AgentTurn) -> float:
        p, c = self.ledger.get(agent_id), turn.confidence
        return {"prestige_x_conf": p * c, "fixed": p * c, "conf_only": c,
                "posthoc": self.scaler(c), "prestige_only": p, "round_robin": 1.0}[self.policy]

    def _pick(self, turns, rnd):
        if self.policy == "round_robin":
            return turns[rnd % len(turns)]
        ws = [self.weight(a, t) for a, t in turns]
        if self.tau is None:
            return turns[max(range(len(ws)), key=ws.__getitem__)]
        return self.rng.choices(turns, weights=[math.exp(w / self.tau) for w in ws], k=1)[0]

    def run(self, problem: dict, update_prestige=True) -> DebateResult:
        transcript, subs = [], defaultdict(list)
        before = view = self.ledger.snapshot()

        for rnd in range(self.rounds):
            if self.parallel and len(self.agents) > 1:
                with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
                    outs = list(ex.map(lambda a: a.propose(problem, transcript, view), self.agents))
            else:
                outs = [a.propose(problem, transcript, view) for a in self.agents]
            turns = [(a.agent_id, t) for a, t in zip(self.agents, outs)]
            for aid, t in turns:
                subs[aid].append(t)
            spk_id, spk = self._pick(turns, rnd)
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
        groups = []   # list of [representative_answer, total_weight]
        for aid, t in last:
            w = self.weight(aid, t)
            for g in groups:
                if answers_match(t.answer, g[0]):
                    g[1] += w
                    break
            else:
                groups.append([t.answer, w])
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
            if norm(r.final_answer) != norm(canon.final_answer):
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


def gemini_population(a):
    tl = None if a.thinking == "none" else a.thinking
    def make():
        return [GeminiAgent(n, n, model=a.model, temperature=a.temperature,
                            thinking_level=tl, log_path=a.log, api=a.api)
                for n in PERSONAS]
    return make


# ===========================================================================
# 7. COMMANDS
# ===========================================================================

def cmd_smoke(a):
    """One call per transport, full traceback. Use this to debug connectivity."""
    import traceback
    p = {"question": "Compute 17 × 23. Give only the integer.", "answer": "391"}
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
                         perturbations=perts)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "mock", "calib", "debate"])
    ap.add_argument("--problems", default=None, help="JSONL file; omit for synthetic arithmetic")
    ap.add_argument("--level", default=None, help="MATH levels to keep, e.g. '4,5' (comma-separated)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5, help="mock only")
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
    a = ap.parse_args()
    _RATE_LIMITER.min_interval = a.stagger
    {"smoke": cmd_smoke, "mock": cmd_mock, "calib": cmd_calib, "debate": cmd_debate}[a.cmd](a)


if __name__ == "__main__":
    main()