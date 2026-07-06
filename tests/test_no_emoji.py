"""Repository-wide no-emoji policy enforcement.

HumaneProxy has a strict no-emoji rule: no emoji characters anywhere in
the repository — source code, output strings, docs, or CI files. Status
markers use bracketed ASCII tags ([OK], [WARN], [SELF-HARM], ...) and doc
tables use plain "Yes"/"No". Emojis previously caused real breakage here
(UnicodeEncodeError on Windows cp1252 terminals) besides the style rule.

This test scans every git-tracked text file and fails on any emoji,
listing file, line, and codepoint. See CONTRIBUTING.md ("Making Changes").
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Emoji & pictograph ranges: Misc Symbols/Dingbats, Emoticons, Transport,
# Supplemental Symbols, Misc Technical, arrows-supplement blocks,
# variation selector 16, and common singletons. Ranges must not overlap
# (each block listed exactly once). Written with escapes so this file
# itself stays emoji-free.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoticons, symbols, transport, flags, supplemental
    "\U00002600-\U000027BF"   # misc symbols + dingbats (incl. check/cross marks)
    "\U00002B00-\U00002BFF"   # misc symbols and arrows (stars etc.)
    "\U00002300-\U000023FF"   # misc technical (watch, hourglass etc.)
    "\U0000FE0F"              # variation selector-16 (emoji presentation)
    "\U00002139"              # information source
    "\U0000203C\U00002049"    # double/interrobang exclamation
    "]"
)

# The one sanctioned exemption: .github/CONTRIBUTING.md's AI-PR title marker
# (U+1F916 robot face) is contributor-policy metadata and stays verbatim.
_ALLOWLIST: dict[str, frozenset[str]] = {
    ".github/CONTRIBUTING.md": frozenset("\U0001F916"),
}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.skip("git not available — cannot enumerate tracked files")
    return result.stdout.split()


def test_no_emojis_in_tracked_files():
    violations: list[str] = []

    for rel_path in _tracked_files():
        path = REPO_ROOT / rel_path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable file

        allowed = _ALLOWLIST.get(rel_path, frozenset())
        for lineno, line in enumerate(lines, 1):
            for char in _EMOJI_RE.findall(line):
                if char in allowed:
                    continue
                violations.append(f"{rel_path}:{lineno}: U+{ord(char):04X}")

    assert not violations, (
        "Emoji characters found (no-emoji policy, see CONTRIBUTING.md):\n  "
        + "\n  ".join(violations)
    )
