"""Tier-1 heuristic classifier — keyword + pattern-based safety check.

Detects self-harm ideation and criminal intent through:
1. Word-boundary keyword matching for known harmful phrases
2. Intent-pattern regex matching for grammatical intent expressions
3. Context reducers to minimize false positives (e.g. "I want to die laughing")

Categories: ``"self_harm"``, ``"criminal_intent"``, ``"safe"``

Scoring is intentionally tuned for **high recall over precision** for
self-harm: a single self-harm keyword match results in a score of 1.0
(critical override).  For criminal intent, scores accumulate from
keyword and pattern matches.
"""

from __future__ import annotations

import re
import unicodedata

from humane_proxy.config import get_config


# ---------------------------------------------------------------------------
# Unicode evasion normalization
# ---------------------------------------------------------------------------
# Prior to this, `classify()` only collapsed whitespace before matching —
# no Unicode normalization at all. That made the keyword/pattern matching
# below trivially bypassable:
#   - zero-width characters inserted mid-keyword ("k\u200bill myself")
#     broke every regex match outright -> classified "safe".
#   - visually-identical non-Latin letters (Cyrillic "у" for Latin "y" in
#     "kill mуself") not only missed the self_harm keyword, but could
#     change which *other* pattern matched instead — flipping the result
#     to the wrong category rather than just missing it.
#
# This is a lightweight, dependency-free normalization pass targeting the
# realistic, demonstrated evasion patterns above — not a general-purpose
# Unicode-confusables library. It runs once per message before any
# matching happens.

# Curated Cyrillic/Greek letters that are visually near-identical to a
# Latin letter and have real-world use as spoofing substitutes. Only
# genuinely confusable single letters are included (e.g. Cyrillic "в" is
# excluded — it reads as Latin "B", not a lowercase letter, so folding it
# would risk unrelated false matches).
_CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ԛ": "q", "ѡ": "w",
    "А": "a", "Е": "e", "О": "o", "Р": "p", "С": "c", "У": "y", "Х": "x",
    "І": "i", "Ѕ": "s", "Ј": "j",
    # Greek -> Latin
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "κ": "k", "ι": "i", "τ": "t",
    "Α": "a", "Ο": "o", "Ρ": "p", "Ν": "v", "Κ": "k", "Ι": "i", "Τ": "t",
}


def _normalize_evasion(text: str) -> str:
    """Neutralize common Unicode-based classifier-evasion tricks.

    Order matters:
      1. NFKC — folds fullwidth/compatibility forms to plain ASCII
         (e.g. the fullwidth "ｋｉｌｌ" -> "kill").
      2. Drop Unicode "Format" category characters (category ``Cf``) —
         zero-width space/joiner/non-joiner, BOM, word joiner, soft
         hyphen, etc. These render invisibly but split regex matches when
         inserted mid-word.
      3. Fold curated Cyrillic/Greek confusables to Latin (see
         ``_CONFUSABLE_MAP``), then NFKD-decompose and drop combining
         diacritical marks (category ``Mn``) to fold accented Latin
         variants (e.g. "kìll" -> "kill").
      4. Re-apply NFKC to recompose into a stable, minimal form.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = "".join(_CONFUSABLE_MAP.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFKC", text)


def _word_boundary_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a case-insensitive word-boundary pattern for *phrase*."""
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.I)


# ---------------------------------------------------------------------------
# Config-derived state.
#
# These globals are (re)built by :func:`_refresh_config` from the **merged**
# configuration (package defaults -> user humane_proxy.yaml -> HUMANE_PROXY_*
# env vars).  They refresh automatically whenever the cached config object
# changes (e.g. after ``config.reload_config()``), so user keyword lists and
# score overrides are honored — previously this module read only the package
# defaults, once, at import time.
# ---------------------------------------------------------------------------
_cfg_snapshot: dict | None = None

_SELF_HARM_KEYWORD_SCORE: float = 0.7
_CRIMINAL_KEYWORD_SCORE: float = 0.6
_INTENT_PATTERN_SCORE: float = 0.7

_SELF_HARM_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
_CRIMINAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
_CONTEXT_REDUCER_PATTERNS: list[re.Pattern[str]] = []


def _refresh_config() -> None:
    """Rebuild keyword/reducer patterns when the merged config changes."""
    global _cfg_snapshot, _SELF_HARM_KEYWORD_SCORE, _CRIMINAL_KEYWORD_SCORE
    global _INTENT_PATTERN_SCORE, _SELF_HARM_PATTERNS, _CRIMINAL_PATTERNS
    global _CONTEXT_REDUCER_PATTERNS

    cfg = get_config().get("heuristics", {})
    if cfg is _cfg_snapshot:
        return
    _cfg_snapshot = cfg

    _SELF_HARM_KEYWORD_SCORE = cfg.get("self_harm_keyword_score", 0.7)
    _CRIMINAL_KEYWORD_SCORE = cfg.get("criminal_keyword_score", 0.6)
    _INTENT_PATTERN_SCORE = cfg.get("intent_pattern_score", 0.7)

    _SELF_HARM_PATTERNS = [
        (kw.lower(), _word_boundary_pattern(kw.lower()))
        for kw in cfg.get("self_harm_keywords", [])
    ]
    _CRIMINAL_PATTERNS = [
        (kw.lower(), _word_boundary_pattern(kw.lower()))
        for kw in cfg.get("criminal_keywords", [])
    ]
    _CONTEXT_REDUCER_PATTERNS = [
        _word_boundary_pattern(r.lower())
        for r in cfg.get("context_reducers", [])
    ]


# ---------------------------------------------------------------------------
# Intent patterns — regex patterns that detect grammatical intent structures
# (static — not configurable; only their score comes from config)
# ---------------------------------------------------------------------------

# Each entry: (name, category, compiled_pattern)
_INTENT_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "first_person_self_harm",
        "self_harm",
        re.compile(
            r"i\s+(want|need|plan|am going|have decided|intend)\s+to\s+"
            r"(kill\s+myself|hurt\s+myself|end\s+my\s+life|harm\s+myself)",
            re.I,
        ),
    ),
    (
        "self_annihilation",
        "self_harm",
        re.compile(r"(end|take).{0,20}(my life|myself|it all)", re.I),
    ),
    (
        "method_seeking",
        "self_harm",
        re.compile(
            r"(painless|easy|easiest|quick|quickest|fast|fastest|best)"
            r".{0,20}(way|method|how).{0,25}"
            r"(die|kill(ing)?\s+(myself|yourself)|end it|end my life|"
            r"overdose|not wake up)",
            re.I,
        ),
    ),
    (
        "first_person_harm_others",
        "criminal_intent",
        re.compile(
            r"i\s+(want|need|plan|am going|intend)\s+to\s+"
            r"(kill|hurt|harm|poison|attack|murder|kidnap|assault)\s+(?!myself)",
            re.I,
        ),
    ),
    (
        "weapons_synthesis",
        "criminal_intent",
        re.compile(
            r"how\s+(do\s+i|to|can\s+i).{0,30}"
            r"(make|build|synthesize|create|manufacture).{0,20}"
            r"(bomb|weapon|explosive|poison|drug|meth|ricin|sarin|fentanyl)",
            re.I,
        ),
    ),
    (
        "acquisition_criminal",
        "criminal_intent",
        re.compile(
            r"where\s+(can\s+i|do\s+i|to).{0,20}"
            r"(buy|get|acquire|find|obtain).{0,20}"
            r"(gun|weapon|drugs|fentanyl|illegal)",
            re.I,
        ),
    ),
    (
        "harm_others_method",
        "criminal_intent",
        re.compile(
            r"(how|ways?\s+(of|to)|best\s+way|easiest\s+way).{0,25}"
            r"(hurt(ing)?|injur(e|ing)|attack(ing)?|electrocut(e|ing)|"
            r"strangl(e|ing)|stab(bing)?|poison(ing)?)\s+"
            r"(someone|somebody|people|a\s+person|my\s+\w+)"
            r"(?!\W*s?\s*feelings)",
            re.I,
        ),
    ),
    (
        "scam_fraud",
        "criminal_intent",
        re.compile(
            r"(how\s+(do\s+i|to|can\s+i)|help\s+me|teach\s+me).{0,30}"
            r"(scam(ming)?|defraud|phish(ing)?|"
            r"steal\s+(\w+('s)?\s+)?(money|passwords?|identity|credit\s+card))",
            re.I,
        ),
    ),
]

def _distinct_signal_clusters(spans: list[tuple[int, int]]) -> int:
    """Count the number of non-overlapping groups among trigger match spans.

    Overlapping spans (a keyword and an intent pattern firing on the same
    phrase) collapse into one cluster; disjoint spans (separate harmful
    expressions in different parts of the message) count individually.
    """
    if not spans:
        return 0
    ordered = sorted(spans)
    clusters = 1
    current_end = ordered[0][1]
    for start, end in ordered[1:]:
        if start >= current_end:
            clusters += 1
            current_end = end
        else:
            current_end = max(current_end, end)
    return clusters


def classify(text: str) -> tuple[str, float, list[str]]:
    """Run heuristic checks on *text* and return ``(category, score, triggers)``.

    Parameters
    ----------
    text:
        The raw user message to evaluate.

    Returns
    -------
    tuple[str, float, list[str]]
        A category string (``"self_harm"``, ``"criminal_intent"``, or
        ``"safe"``), a risk score clamped to ``[0.0, 1.0]``, and a
        **deduplicated** list of human-readable trigger descriptions.
    """
    # Pick up any config changes (user yaml / env overrides / reloads).
    _refresh_config()

    # Guard: empty / whitespace-only input.
    if not text or not text.strip():
        return "safe", 0.0, []

    # Neutralize Unicode-based evasion (zero-width chars, homoglyphs,
    # fullwidth forms, diacritics) before any keyword/pattern matching.
    text = _normalize_evasion(text)

    # Normalize whitespace: collapse runs of spaces/tabs/newlines into a
    # single space.  Defeats evasion tricks like embedded newlines.
    text = re.sub(r"\s+", " ", text).strip()

    score: float = 0.0
    category: str = "safe"
    seen: set[str] = set()                    # O(1) dedup
    triggers: list[str] = []                  # ordered output
    spans: list[tuple[int, int]] = []         # match positions for reducer gating

    # --- Self-harm keyword scan ---
    for keyword, pattern in _SELF_HARM_PATTERNS:
        if match := pattern.search(text):
            tag = f"self_harm_keyword:{keyword}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                spans.append(match.span())
                score += _SELF_HARM_KEYWORD_SCORE
                category = "self_harm"
            if score >= 1.0:
                break

    # --- Criminal keyword scan ---
    for keyword, pattern in _CRIMINAL_PATTERNS:
        if match := pattern.search(text):
            tag = f"criminal_keyword:{keyword}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                spans.append(match.span())
                score += _CRIMINAL_KEYWORD_SCORE
                if category != "self_harm":
                    category = "criminal_intent"
            if score >= 1.0:
                break

    # --- Intent pattern scan ---
    for name, pat_category, pattern in _INTENT_PATTERNS:
        if match := pattern.search(text):
            tag = f"intent_pattern:{name}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                spans.append(match.span())
                score += _INTENT_PATTERN_SCORE
                # Self-harm patterns upgrade the category.
                if pat_category == "self_harm":
                    category = "self_harm"
                elif category == "safe":
                    category = pat_category
            if score >= 1.0:
                break

    # --- Context reducer check ---
    # Reducers activate only when every trigger stems from ONE expression in
    # the text (their match spans all overlap).  A keyword and an intent
    # pattern routinely co-fire on the same phrase ("how to make a bomb in
    # minecraft" trips both), and counting that as two independent signals
    # used to lock reducers out of exactly the false positives they exist
    # for.  Two or more *separate* harmful expressions still disable
    # reduction — that is genuine concern, not stray context.
    if score > 0.0 and _distinct_signal_clusters(spans) == 1:
        for reducer_pattern in _CONTEXT_REDUCER_PATTERNS:
            if reducer_pattern.search(text):
                score *= 0.1
                triggers.append("context_reduced")
                category = "safe"
                break

    # --- Self-harm critical override ---
    # If category is still self_harm after context reduction, force score
    # to 1.0.  Every self-harm signal matters.
    if category == "self_harm":
        score = 1.0

    score = min(score, 1.0)

    return category, score, triggers
