"""Characters a Japanese company document may contain; OCR output is masked to this set."""

from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=None)
def japanese(extra_file=None):
    """Windows Japanese (cp932 = JIS X 0208 + NEC/IBM extensions such as 髙 﨑 ① ㈱ ㎡) and ASCII.

    `extra_file` (default: $PDF_PARSER_EXTRA_CHARS) adds characters, e.g. every character
    found in the company DOCX corpus. Simplified-only forms such as 值 说 们 are excluded.
    """
    chars = set()
    for code in range(0x20, 0x30000):
        character = chr(code)
        try:
            character.encode("cp932")
        except UnicodeEncodeError:
            continue
        chars.add(character)
    extra_file = extra_file or os.environ.get("PDF_PARSER_EXTRA_CHARS")
    if extra_file:
        chars.update(c for c in Path(extra_file).read_text(encoding="utf-8") if c.isprintable())
    return frozenset(chars)


def regex_class(chars):
    """A regex character class for `chars` (plus newline) in syntax xgrammar and Python share."""
    special = set("\\]^-[")
    return "[" + "".join("\\" + c if c in special else c for c in sorted(chars)) + "\\n]"


def is_kanji(character):
    code = ord(character)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x2FFFF or character in "々〆〇")


def is_kana(character):
    code = ord(character)
    return 0x3041 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F
