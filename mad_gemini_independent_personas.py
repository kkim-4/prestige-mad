#!/usr/bin/env python3
"""
mad_gemini.py — prestige-weighted, confidence-scored multi-agent debate, all in one file.

Setup:
    pip install google-genai pydantic sympy
    export GEMINI_API_KEY="..."

Use:
    python3 mad_gemini.py smoke                     # 1 call per transport, full errors
    python3 mad_gemini.py mock                      # mechanism sanity check, no API
    python3 mad_gemini.py calib --n 20              # does stated confidence track correctness?
    python3 mad_gemini.py debate --problems math.jsonl --n 15

Independence options (new):
    --board full|blind      blind hides others' confidence and prestige from the transcript
    --population same|mixed mixed = personas spread across different models / thinking levels
    --panel SPEC            custom panel, e.g.
                            "careful@gemini-3.8-flash:high,fast@gemini-3.1-flash-lite:low:1.0"
                            format per agent: persona[@model][:thinking][:temperature]

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
# Field order matters: structured output is generated in this order, so the
# agent names its method and reasoning, commits to an answer, says whether it
# changed and why, names its biggest doubt, and only THEN states a confidence.
# No field has a default value: the Gemini response_schema rejects defaults.

class AgentTurn(BaseModel):
    method: str = Field(
        description="The approach you used, in a few words (e.g. 'complementary counting', "
                    "'coordinates', 'substitution check').")
    rationale: str = Field(
        description="Your reasoning in at most 5 sentences: the key steps, the check you ran, "
                    "and (after round 0) where you agree or disagree with other agents.")
    answer: str = Field(description="Final answer only, no units or explanation.")
    changed_answer: bool = Field(
        description="True only if this answer differs from your own previous-round answer. "
                    "Always false on your first attempt.")
    change_reason: str = Field(
        description="If changed_answer is true: the specific step (yours or another agent's, "
                    "verified by you) that made you change. Otherwise an empty string.")
    key_uncertainty: str = Field(
        description="The single thing most likely to make `answer` wrong.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Honest probability that `answer` is correct, using the confidence bands "
                    "in your instructions. Scored with a proper scoring rule against ground "
                    "truth after the problem closes.")

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
# 3. PERSONAS & PROMPTS — built to make agents work independently
# ===========================================================================
# Each persona differs in HOW it solves (method), HOW it verifies (check), and
# HOW it treats the transcript (stance), not just in temperament. Different
# methods make different mistakes, which is what gives a vote or a debate
# something to work with.

PERSONAS = {
    "careful": {
        "title": "the methodical deriver",
        "method": (
            "Restate the givens and exactly what is being asked, in your own words. Then "
            "derive the answer forward in small, explicit steps, keeping exact forms "
            "(fractions, radicals, pi) until the very end. Do not skip steps you could do in "
            "your head: slips happen in the steps you skip."),
        "check": (
            "Substitute your answer back into the original conditions, or recompute the "
            "single most error-prone step by a different arithmetic route. If the check "
            "fails, find the step that broke before you answer."),
        "stance": (
            "Treat every answer in the transcript as a hypothesis to test against your own "
            "derivation, never as a shortcut. If someone disagrees with you, find the first "
            "line where your work and theirs diverge and decide that line on its merits."),
        "watch": (
            "Arithmetic and sign slips, dropped cases, and quietly changing the problem "
            "while restating it."),
    },
    "fast": {
        "title": "the estimator",
        "method": (
            "Before computing anything, write down a rough estimate or bound for the answer: "
            "its sign, size, parity or range, and anything else visible at a glance. Then "
            "take the most direct route you know, using standard results and shortcuts where "
            "their conditions genuinely hold."),
        "check": (
            "Confirm the final answer falls inside your estimate and satisfies an easy "
            "special case (a small n, a degenerate shape, a simple value). If the estimate "
            "and the computation disagree, one of them is wrong: find out which."),
        "stance": (
            "Commit to your own number. If the transcript disagrees, say so plainly and test "
            "whether their answer passes your estimate. Adopt it only if it passes and you "
            "can see where your own route went wrong."),
        "watch": (
            "Applying a formula outside its conditions, and missing a constraint stated in "
            "the problem."),
    },
    "skeptic": {
        "title": "the adversarial checker",
        "method": (
            "First solve the problem yourself by whatever method you trust. Then turn on "
            "your own answer and try to break it: test boundary and degenerate cases, confirm "
            "every condition in the problem was used, and ask whether a nearby answer (off by "
            "one, a missing factor of 2, ordered vs unordered counting) would look just as "
            "plausible."),
        "check": (
            "An answer survives only if you tried at least one concrete way it could be "
            "wrong and it held up. Name that attempt in your rationale."),
        "stance": (
            "For each answer in the transcript, look for the flaw before looking for the "
            "merit. Agree only after an honest attempt to find an error has failed. Being "
            "the lone dissenter is acceptable and often valuable; being talked into "
            "agreement without a verified reason is not."),
        "watch": (
            "Accepting a confident, fluent argument without checking it, including your "
            "own."),
    },
    "teacher": {
        "title": "the careful reader",
        "method": (
            "Start with the wording. Identify definitions, units, domain restrictions "
            "(integers? positive? distinct?), and exactly which quantity and form is "
            "requested (simplest form, interval notation, a given base, degrees vs radians, "
            "a sum vs a count). Then solve with the simplest clean approach, as a worked "
            "solution a student could follow line by line."),
        "check": (
            "Reread the question after solving and confirm your answer is the quantity "
            "asked for, in the form asked for. Many wrong answers are correct answers to a "
            "slightly different question."),
        "stance": (
            "Judge other agents' reasoning by whether it answers the question as written and "
            "whether each step follows from the last. Point out when another agent answered "
            "a different question or gave the right value in the wrong form."),
        "watch": "Misreading the problem, and answering in the wrong format.",
    },
    "contrarian": {
        "title": "the alternate-route solver",
        "method": (
            "Deliberately solve by a different method from the most obvious one: "
            "complementary counting instead of direct counting, coordinates instead of "
            "synthetic geometry, algebra instead of a memorized formula, enumerating small "
            "cases to find a pattern, or working backward from the answer's form. Carry "
            "that alternative route all the way to a final value."),
        "check": (
            "If a second route is quick, do it too. Two different methods that agree are "
            "much stronger evidence than one method done twice."),
        "stance": (
            "When the transcript has converged, you are most useful: re-derive by a "
            "different route and report what you actually get. Agree only if your "
            "independent route lands on the same answer; if it does not, say which result "
            "you trust and why."),
        "watch": (
            "Groupthink, and also the opposite error of disagreeing for its own sake. Your "
            "job is independent verification, not contrarian answers."),
    },
}


def render_persona(key: str) -> str:
    p = PERSONAS[key]
    return (f"You are {key.upper()}, {p['title']}.\n\n"
            f"HOW YOU SOLVE:\n{p['method']}\n\n"
            f"HOW YOU CHECK:\n{p['check']}\n\n"
            f"HOW YOU TREAT OTHER AGENTS:\n{p['stance']}\n\n"
            f"YOUR TYPICAL FAILURE TO GUARD AGAINST:\n{p['watch']}")


BOARD_NOTE = {
    "full": (
        "Each round you see a transcript of the agent chosen to speak: their answer, method, "
        "reasoning, stated confidence, and PRESTIGE (0-1), a running record of how well that "
        "agent's stated confidence has matched actual correctness on past problems."),
    "blind": (
        "Each round you see a transcript of the agent chosen to speak: their answer, method "
        "and reasoning. Their confidence and track record are hidden on purpose: judge the "
        "argument, not the reputation."),
}

SYSTEM = """{persona_block}

=== THE GROUP ===
You are one agent in a small group solving a math problem over several rounds.
{board_note}
After the last round, the group's answer is decided by a vote over every agent's
final answer.

=== WORKING INDEPENDENTLY ===
1. Solve it yourself first. In every round, work the problem from the statement
   using YOUR method (above) before you weigh anything in the transcript. The
   transcript is evidence to examine, not an answer key.
2. The other agents run on similar models and share many of your blind spots.
   Several agents agreeing is much weaker evidence than it looks: a shared answer
   can be a shared mistake.
3. Change your answer only for a concrete reason: a specific step in your own
   earlier work that you now see is wrong, or a specific argument in the
   transcript that you have checked yourself. "Others agree", "higher prestige"
   or "higher confidence" is never a reason on its own. If you change, set
   changed_answer=true and name that step in change_reason.
4. If you keep an answer that others dispute, say in your rationale where you
   believe their reasoning fails.
5. Dissent is useful. The final answer is a vote, and an honest minority answer
   helps the group more than a copied one. Being the only agent with a different
   answer is fine if your work supports it.
6. Stay in your role: use your own method even when another method has already
   been posted.

=== CONFIDENCE ===
Your stated confidence is scored against the true answer with a proper scoring
rule after the problem closes. Being confident and wrong costs you prestige; being
right but hedged costs you some too. You maximise your expected score only by
reporting the probability you actually believe. Prestige decides how much say you
get in future rounds. Use these bands:
  0.97-0.99  solved AND confirmed by an independent check (a second method,
             substitution into the original conditions, or full enumeration),
             and you are sure you answered exactly what was asked
  0.85-0.95  one clean derivation plus a partial check
  0.60-0.80  a derivation you believe but did not verify, or one shaky step
  0.30-0.60  choosing between candidate answers
  below 0.30 mostly a guess
Other agents agreeing does NOT move you into a higher band unless their reasoning
let you verify a step you could not verify alone."""

USER = """PROBLEM:
{question}
{standings}
YOUR PREVIOUS ANSWER:
{own_prev}

STEP 1. Work the problem yourself with your own method, from the statement, before
reading the transcript below.

TRANSCRIPT SO FAR (what the chosen speakers said; examine it critically):
{board}

STEP 2. Compare your independent result with the transcript. Keep or change your
answer according to the rules in your instructions, then respond in the required
JSON format."""


def render_system(persona: str, board: str = "full") -> str:
    return SYSTEM.format(persona_block=render_persona(persona), board_note=BOARD_NOTE[board])


def render_user(agent_id: str, problem: dict, transcript: list, prestige: dict,
                own_prev=None, board: str = "full") -> str:
    lines = []
    for t in transcript:
        who = t["speaker"] + (" (you)" if t["speaker"] == agent_id else "")
        if board == "full":
            head = (f"[round {t['round']}] {who} (prestige {t['speaker_prestige']:.2f}, "
                    f"confidence {t['confidence']:.2f}) -> {t['answer']}")
        else:
            head = f"[round {t['round']}] {who} -> {t['answer']}"
        lines.append(head)
        if t.get("method"):
            lines.append(f"   method: {t['method']}")
        lines.append(f"   reasoning: {t['rationale']}")
    board_txt = "\n".join(lines) or "(no one has spoken yet)"

    if board == "full" and prestige:
        standings = ("\nCURRENT PRESTIGE: "
                     + ", ".join(f"{a}: {p:.2f}" for a, p in sorted(prestige.items())) + "\n")
    else:
        standings = ""

    if own_prev is None:
        own = "(none; this is your first attempt)"
    else:
        own = (f"{own_prev.answer} (your confidence {own_prev.confidence:.2f}; "
               f"method: {own_prev.method or 'n/a'})")

    return USER.format(question=problem["question"], standings=standings,
                       own_prev=own, board=board_txt)


# ===========================================================================
# 4. AGENTS — interface: propose(problem, transcript, prestige, own_prev) -> AgentTurn
# ===========================================================================

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
    board: str = "full"                 # "full" | "blind"

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
        self.system = render_system(self.persona, self.board)

    def propose(self, problem: dict, transcript: list, prestige: dict,
                own_prev: AgentTurn = None) -> AgentTurn:
        prompt = render_user(self.agent_id, problem, transcript, prestige, own_prev, self.board)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                _RATE_LIMITER.wait()
                t0 = time.time()
                text = self._call(prompt)
                turn = AgentTurn.model_validate_json(text).clipped()
                self._log({"agent": self.agent_id, "persona": self.persona,
                           "model": self.model, "thinking": self.thinking_level,
                           "temperature": self.temperature, "api": self.api,
                           "board": self.board, "ts": time.time(),
                           "q": problem["question"], "truth": problem.get("answer"),
                           "round": len(transcript),
                           "own_prev": own_prev.answer if own_prev is not None else None,
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

    def propose(self, problem, transcript, prestige, own_prev=None) -> AgentTurn:
        correct = self.rng.random() < self.accuracy
        d = problem.get("distractors") or [str(problem["answer"]) + "0", "-1"]
        answer = problem["answer"] if correct else (d[0] if self.rng.random() < 0.6 else self.rng.choice(d))
        p = self.accuracy if correct else 1 - self.accuracy
        if transcript:
            last = transcript[-1]
            if self.rng.random() < self.conformity * last["speaker_prestige"] * last["confidence"]:
                answer, p = last["answer"], max(p, last["confidence"] * 0.8)
        c = min(0.99, max(0.01, p + self.conf_bias + self.rng.gauss(0, 0.05)))
        changed = own_prev is not None and norm(own_prev.answer) != norm(answer)
        return AgentTurn(method="mock", rationale="mock", answer=str(answer),
                         changed_answer=changed, change_reason="mock" if changed else "",
                         key_uncertainty="", confidence=c)


# ===========================================================================
# 5. ORCHESTRATOR — one problem = one debate = N rounds
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
            # each agent sees its OWN previous answer, so it can re-solve and
            # then decide explicitly whether to keep or change it
            prev = {a.agent_id: (subs[a.agent_id][-1] if subs[a.agent_id] else None)
                    for a in self.agents}
            if self.parallel and len(self.agents) > 1:
                with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
                    outs = list(ex.map(
                        lambda a: a.propose(problem, transcript, view, own_prev=prev[a.agent_id]),
                        self.agents))
            else:
                outs = [a.propose(problem, transcript, view, own_prev=prev[a.agent_id])
                        for a in self.agents]
            turns = [(a.agent_id, t) for a, t in zip(self.agents, outs)]
            for aid, t in turns:
                subs[aid].append(t)
            spk_id, spk = self._pick(turns, rnd)
            transcript.append({"round": rnd, "speaker": spk_id, "answer": spk.answer,
                               "confidence": spk.confidence, "rationale": spk.rationale,
                               "method": spk.method,
                               "speaker_prestige": view.get(spk_id, self.ledger.prior)})

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
# 6. INDEPENDENCE DIAGNOSTICS — are the agents actually thinking separately?
# ===========================================================================

def cluster_answers(answers) -> list:
    """Group equivalent answers; returns a list of groups (lists of answers)."""
    groups = []
    for a in answers:
        for g in groups:
            if answers_match(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    return groups


def debate_dynamics(res: DebateResult, truth) -> dict:
    """Counts for one debate:
    r0_split     round-0 answers were not unanimous (agents disagreed before seeing anyone)
    final_split  final answers were not unanimous
    agent_rounds agent-turns after round 0 (opportunities to switch)
    switches     turns where an agent's answer differed from its own previous answer
    herd         switches that landed on an answer already posted in the transcript
    good / bad   switches wrong->right / right->wrong"""
    ids = list(res.submissions)
    r0 = [res.submissions[a][0].answer for a in ids]
    fin = [res.submissions[a][-1].answer for a in ids]
    d = {"r0_split": int(len(cluster_answers(r0)) > 1),
         "final_split": int(len(cluster_answers(fin)) > 1),
         "agent_rounds": 0, "switches": 0, "herd": 0, "good": 0, "bad": 0}
    for a in ids:
        ts = res.submissions[a]
        for r in range(1, len(ts)):
            d["agent_rounds"] += 1
            prev, cur = ts[r - 1].answer, ts[r].answer
            if answers_match(prev, cur):
                continue
            d["switches"] += 1
            if any(answers_match(cur, t["answer"]) for t in res.transcript[:r]):
                d["herd"] += 1
            pv, cv = answers_match(prev, truth), answers_match(cur, truth)
            if cv and not pv:
                d["good"] += 1
            elif pv and not cv:
                d["bad"] += 1
    return d


def summarize_dynamics(ds: list) -> dict:
    n = max(1, len(ds))
    tot = {k: sum(d[k] for d in ds) for k in ("r0_split", "final_split", "agent_rounds",
                                                "switches", "herd", "good", "bad")}
    return {"r0_split_rate": tot["r0_split"] / n,
            "final_split_rate": tot["final_split"] / n,
            "switch_rate": tot["switches"] / max(1, tot["agent_rounds"]),
            "herd_share": tot["herd"] / max(1, tot["switches"]),
            "switch_good": tot["good"],
            "switch_bad": tot["bad"]}


# ===========================================================================
# 7. EVALUATION — accuracy + stability under three perturbations
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
    dynamics: dict = field(default_factory=dict)

    def row(self) -> str:
        dy = self.dynamics or {}
        return (f"{self.policy:16s} acc={self.accuracy:.2f} flip={self.flip_rate:.2f} "
                f"C->W={self.flip_c2w:.2f} W->C={self.flip_w2c:.2f} | "
                f"order={self.flip_by_kind['order']:.2f} drop={self.flip_by_kind['drop']:.2f} "
                f"fresh={self.flip_by_kind['fresh']:.2f} | "
                f"r0split={dy.get('r0_split_rate', 0):.2f} "
                f"finalsplit={dy.get('final_split_rate', 0):.2f} "
                f"switch={dy.get('switch_rate', 0):.2f} herd={dy.get('herd_share', 0):.2f} "
                f"good/bad={dy.get('switch_good', 0)}/{dy.get('switch_bad', 0)}")


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
    dyns = []

    for i, p in enumerate(test):
        canon = orch.run(p, update_prestige=False)
        n_correct += canon.correct
        dyn = debate_dynamics(canon, p["answer"])
        dyns.append(dyn)

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
                  f"r0split={dyn['r0_split']} switches={dyn['switches']} herd={dyn['herd']} "
                  f"prestige={ {a: round(v,2) for a,v in ledger.snapshot().items()} }", flush=True)

    n_pert = sum(kind_n.values())
    return Metrics(policy=policy, n_test=len(test),
                   accuracy=n_correct / max(1, len(test)),
                   flip_rate=flips / max(1, n_pert),
                   flip_c2w=c2w / max(1, n_pert), flip_w2c=w2c / max(1, n_pert),
                   flip_by_kind={k: kind_flips[k] / max(1, kind_n[k]) for k in kind_n},
                   prestige_final=ledger.snapshot(),
                   mean_score={a: ledger.mean_score(a) for a in ids},
                   ece={a: ledger.ece(a) for a in ids},
                   dynamics=summarize_dynamics(dyns))


# ===========================================================================
# 8. PROBLEMS & POPULATIONS
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


# Default mixed panel: same five personas, spread over two models and two
# thinking levels so agents differ in real skill, not just in wording.
# Override with --panel. Check that each model/thinking combo works with
# `smoke --model ... --thinking ...` before a long run.
DEFAULT_MIXED_PANEL = ("careful@gemini-3.8-flash:high,"
                       "fast@gemini-3.1-flash-lite:low,"
                       "skeptic@gemini-3.8-flash:low,"
                       "teacher@gemini-3.1-flash-lite:high,"
                       "contrarian@gemini-3.8-flash:low")


def parse_panel(spec: str, default_model, default_thinking, default_temp) -> list:
    """'persona[@model][:thinking][:temperature],...' -> list of agent configs.
    thinking is low|high|none; omitted parts fall back to the CLI defaults.
    A persona used twice gets a numbered id (skeptic, skeptic2, ...)."""
    out, seen = [], defaultdict(int)
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        parts = item.split(":")
        persona, _, model = parts[0].partition("@")
        persona = persona.strip()
        if persona not in PERSONAS:
            sys.exit(f"--panel: unknown persona '{persona}'. Choose from {sorted(PERSONAS)}.")
        thinking = parts[1].strip() if len(parts) > 1 and parts[1].strip() else default_thinking
        if thinking not in ("low", "high", "none"):
            sys.exit(f"--panel: thinking must be low|high|none, got '{thinking}' in '{item}'.")
        try:
            temp = float(parts[2]) if len(parts) > 2 and parts[2].strip() else default_temp
        except ValueError:
            sys.exit(f"--panel: bad temperature in '{item}'.")
        seen[persona] += 1
        aid = persona if seen[persona] == 1 else f"{persona}{seen[persona]}"
        out.append({"agent_id": aid, "persona": persona,
                    "model": model.strip() or default_model,
                    "thinking": None if thinking == "none" else thinking,
                    "temperature": temp})
    if len(out) < 2:
        sys.exit("--panel needs at least two agents.")
    return out


def gemini_population(a):
    if a.panel:
        spec = a.panel
    elif a.population == "mixed":
        spec = DEFAULT_MIXED_PANEL
    else:
        spec = ",".join(PERSONAS)          # every persona on --model / --thinking
    cfgs = parse_panel(spec, a.model, a.thinking, a.temperature)

    def make():
        return [GeminiAgent(c["agent_id"], c["persona"], model=c["model"],
                            temperature=c["temperature"], thinking_level=c["thinking"],
                            log_path=a.log, api=a.api, board=a.board)
                for c in cfgs]
    return make


def describe_panel(agents) -> str:
    return "\n".join(f"  {ag.agent_id:12s} persona={ag.persona:10s} model={ag.model} "
                     f"thinking={ag.thinking_level} temp={ag.temperature}" for ag in agents)


# ===========================================================================
# 9. COMMANDS
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
                             log_path=None, api=api, max_retries=1, board=a.board)
            t = ag.propose(p, [], {"careful": 0.5})
            ok = "CORRECT" if norm(t.answer) == "391" else "WRONG"
            print(f"SUCCESS [{ok}] answer={t.answer} confidence={t.confidence:.2f} method={t.method}")
            print(f"        rationale: {t.rationale[:100]}")
        except Exception:
            traceback.print_exc()
    print("\nIf legacy works and interactions doesn't, just use --api legacy (the default).")


def cmd_prompts(a):
    """Print the system and user prompts each persona would see. No API calls."""
    p = {"question": "How many positive divisors does 36 have?", "answer": "9"}
    prestige = {k: 0.5 for k in PERSONAS}
    prev = AgentTurn(method="prime factorization", rationale="36 = 2^2 * 3^2, so (2+1)(2+1).",
                     answer="9", changed_answer=False, change_reason="",
                     key_uncertainty="none", confidence=0.95)
    transcript = [{"round": 0, "speaker": "fast", "answer": "8", "confidence": 0.9,
                   "rationale": "Listed divisors quickly.", "method": "listing",
                   "speaker_prestige": 0.5}]
    keys = [a.persona] if a.persona else list(PERSONAS)
    for k in keys:
        print("=" * 78)
        print(f"SYSTEM ({k}, board={a.board})\n")
        print(render_system(k, a.board))
        print("-" * 78)
        print("USER, round 0\n")
        print(render_user(k, p, [], prestige, None, a.board))
        print("-" * 78)
        print("USER, round 1\n")
        print(render_user(k, p, transcript, prestige, prev, a.board))


def cmd_mock(a):
    print(f"{'policy':16s}  acc  flip  C->W  W->C | order  drop fresh | r0split switch herd"
          f"   (mean of {a.seeds} seeds)")
    for pol in a.policies.split(","):
        ms = [evaluate(mock_population(s), synthetic(a.n, s), pol, a.rounds, alpha=a.alpha,
                       rule=a.rule, stake=a.stake, tau=a.tau, seed=s) for s in range(a.seeds)]
        avg = lambda f: sum(f(m) for m in ms) / len(ms)
        print(f"{pol:16s} {avg(lambda m: m.accuracy):.2f}  {avg(lambda m: m.flip_rate):.2f}  "
              f"{avg(lambda m: m.flip_c2w):.2f}  {avg(lambda m: m.flip_w2c):.2f} | "
              f"{avg(lambda m: m.flip_by_kind['order']):.2f}  {avg(lambda m: m.flip_by_kind['drop']):.2f}  "
              f"{avg(lambda m: m.flip_by_kind['fresh']):.2f} | "
              f"{avg(lambda m: m.dynamics['r0_split_rate']):.2f}    "
              f"{avg(lambda m: m.dynamics['switch_rate']):.2f}   "
              f"{avg(lambda m: m.dynamics['herd_share']):.2f}")
        print("   prestige:", {k: round(v, 2) for k, v in ms[-1].prestige_final.items()})


def _levels(a):
    return {int(x) for x in a.level.split(",")} if a.level else None


def cmd_calib(a):
    """MILESTONE 1: single agent, no debate. Does stated confidence predict correctness?
    Also the cheapest test of persona independence: how often do round-0 answers differ?"""
    probs = load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0)
    agents = gemini_population(a)()
    ids = [x.agent_id for x in agents]
    led = PrestigeLedger(ids)
    lvl_str = f"levels={sorted(_levels(a))}" if a.level else "levels=all"
    print(f"Running {len(probs)} problems x {len(agents)} agents "
          f"({len(probs)*len(agents)} calls, {a.api or 'legacy'} transport, "
          f"board={a.board}, {lvl_str})\n{describe_panel(agents)}\n")
    splits = 0
    agree = defaultdict(int)          # (agent_i, agent_j) -> #problems with equivalent answers
    for i, p in enumerate(probs):
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            turns = list(ex.map(lambda ag: ag.propose(p, [], led.snapshot()), agents))
        marks = []
        for ag, t in zip(agents, turns):
            ok = answers_match(t.answer, p["answer"])
            led.update(ag.agent_id, t.confidence, ok)
            marks.append(f"{ag.agent_id[:4]}{'+' if ok else '-'}{t.confidence:.2f}")
        answers = [t.answer for t in turns]
        split = len(cluster_answers(answers)) > 1
        splits += split
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                if answers_match(answers[x], answers[y]):
                    agree[(ids[x], ids[y])] += 1
        print(f"[{i+1}/{len(probs)}] {'SPLIT' if split else '     '} truth={p['answer']:<8} "
              + "  ".join(marks), flush=True)

    print(f"\n{'agent':12s} {'acc':>5s} {'meanconf':>9s} {'1-brier':>8s} {'ECE':>5s}")
    for ag in agents:
        r = led.records[ag.agent_id]
        acc = sum(y for _, y, _ in r) / len(r)
        mc = sum(c for c, _, _ in r) / len(r)
        print(f"{ag.agent_id:12s} {acc:5.2f} {mc:9.2f} {led.mean_score(ag.agent_id):8.2f} "
              f"{led.ece(ag.agent_id):5.2f}")

    n = max(1, len(probs))
    print(f"\nIndependence: agents disagreed on {splits}/{len(probs)} problems "
          f"({splits / n:.0%}).")
    print("Pairwise agreement (share of problems with equivalent answers):")
    print(" " * 12 + "".join(f"{x[:6]:>8s}" for x in ids))
    for x in ids:
        row = []
        for y in ids:
            if x == y:
                row.append(f"{'-':>8s}")
            else:
                k = (x, y) if (x, y) in agree else (y, x)
                row.append(f"{agree.get(k, 0) / n:8.2f}")
        print(f"{x[:12]:12s}" + "".join(row))
    print("\nRead this: if acc is ~1.00 everywhere, the problems are too easy to show\n"
          "anything — use harder ones. If acc is far below meanconf everywhere (high ECE),\n"
          "confidence isn't carrying signal; fix elicitation before running debates.\n"
          "If agents almost never disagree, the panel has nothing for a policy to decide.")


def cmd_debate(a):
    probs = load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0)
    print("Panel:\n" + describe_panel(gemini_population(a)()) + f"\nboard={a.board}")
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
    ap.add_argument("cmd", choices=["smoke", "prompts", "mock", "calib", "debate"])
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
    ap.add_argument("--population", default="same", choices=["same", "mixed"],
                    help="same: every persona on --model/--thinking. mixed: DEFAULT_MIXED_PANEL")
    ap.add_argument("--panel", default=None,
                    help="custom panel, overrides --population: "
                         "'persona[@model][:thinking][:temp],...'")
    ap.add_argument("--board", default="full", choices=["full", "blind"],
                    help="blind hides other agents' confidence and prestige from the transcript")
    ap.add_argument("--persona", default=None, help="prompts only: show one persona")
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
    {"smoke": cmd_smoke, "prompts": cmd_prompts, "mock": cmd_mock,
     "calib": cmd_calib, "debate": cmd_debate}[a.cmd](a)


if __name__ == "__main__":
    main()