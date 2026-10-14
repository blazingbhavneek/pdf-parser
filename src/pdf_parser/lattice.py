"""Character lattices: OCR readers propose characters, larger models may only choose among them.

A lattice is a list of `Char`. `segments()` turns it into locked text runs and slots
(tuples of alternatives); `template()` shows slots as ⟦A|B⟧ for a prompt and `regex()` builds
the decoder constraint, so a model filling the slots can never write anything else.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math

OPEN, BAR, CLOSE = "⟦", "|", "⟧"
_SPECIAL = set("\\.^$|?*+()[]{}")


@dataclass
class Char:
    c: str  # chosen character ("" = a reader saw nothing here)
    p: float  # confidence of `c`, 0..1
    alts: dict = field(default_factory=dict)  # candidate -> probability, always includes `c`
    agreed: int = 1  # how many readers chose `c`

    def __post_init__(self):
        self.alts.setdefault(self.c, self.p)


def text(chars):
    return "".join(char.c for char in chars)


def locked(string, p=1.0):
    return [Char(c, p, {c: p}) for c in string]


def from_reader(chars, allowed=None):
    """Chars from the model server's recognizer output ({"c", "p", "alts": [[c, p], ...]})."""
    result = []
    for item in chars:
        alts = {c: p for c, p in item["alts"] if allowed is None or c in allowed}
        result.append(Char(item["c"], item["p"], alts))
    return result


def join_lines(lines, newline=False):
    """Concatenate line lattices: newlines for code, else a space only between Latin words."""
    result = []
    for line in lines:
        if result and line:
            if newline:
                result.append(Char("\n", 1.0))
            elif result[-1].c.isascii() and result[-1].c.isalnum() and line[0].c.isascii() and line[0].c.isalnum():
                result.append(Char(" ", 1.0))
        result.extend(line)
    return result


def join_text(strings, newline=False):
    return text(join_lines([locked(s.strip()) for s in strings], newline))


def unwrap(chars):
    """Drop line breaks inside a paragraph (a space only between Latin words), trim the ends."""
    result = []
    for index, char in enumerate(chars):
        if char.c != "\n":
            result.append(char)
            continue
        after = next((c for c in chars[index + 1:] if c.c != "\n"), None)
        if result and after and result[-1].c.isascii() and result[-1].c.isalnum() \
                and after.c.isascii() and after.c.isalnum():
            result.append(Char(" ", char.p))
    return strip(result)


def strip(chars):
    start, end = 0, len(chars)
    while start < end and chars[start].c.isspace():
        start += 1
    while end > start and chars[end - 1].c.isspace():
        end -= 1
    return chars[start:end]


def from_logprobs(content, tokens, allowed=None):
    """One Char per character of `content` from vLLM chat logprobs.

    Tokens may hold several characters or only part of one (UTF-8 bytes); alternatives come
    from top tokens of the same length that decode on their own.
    """
    chars, pending, pending_p = [], b"", 1.0
    for token in tokens:
        raw = bytes(token["bytes"]) if token.get("bytes") is not None else token["token"].encode()
        p = math.exp(token["logprob"])
        if pending:
            raw, p = pending + raw, min(p, pending_p)
        try:
            piece = raw.decode("utf-8")
        except UnicodeDecodeError:
            pending, pending_p = raw, p
            continue
        alternatives = {}
        if not pending:
            for alt in token.get("top_logprobs") or []:
                alt_raw = bytes(alt["bytes"]) if alt.get("bytes") is not None else alt["token"].encode()
                try:
                    candidate = alt_raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if len(candidate) == len(piece) and candidate != piece:
                    alternatives[candidate] = math.exp(alt["logprob"])
        pending, pending_p = b"", 1.0
        for index, character in enumerate(piece):
            alts = {character: p}
            for candidate, q in alternatives.items():
                other = candidate[index]
                if other != character and (allowed is None or other in allowed):
                    alts[other] = max(alts.get(other, 0.0), q)
            chars.append(Char(character, p, alts))
    if text(chars) != content:
        # The server's tokens did not reproduce its text: keep the text, mean confidence, no alternatives.
        mean = math.exp(sum(t["logprob"] for t in tokens) / len(tokens)) if tokens else 0.0
        return [Char(c, mean, {c: mean}) for c in content]
    return chars


def pairs(a, b):
    """Align strings a and b: (i, j) for matched or substituted positions, None for a gap."""
    result = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "delete":
            result += [(i, None) for i in range(i1, i2)]
        elif tag == "insert":
            result += [(None, j) for j in range(j1, j2)]
        else:  # equal / replace: pair positionally, the rest are gaps
            n = min(i2 - i1, j2 - j1)
            result += [(i1 + k, j1 + k) for k in range(n)]
            result += [(i, None) for i in range(i1 + n, i2)]
            result += [(None, j) for j in range(j1 + n, j2)]
    return result


def merge(primary, secondary):
    """The primary reader's lattice, confirmed or contested by a second reader aligned to it.

    Agreement raises confidence (noisy-or); a disagreement always becomes a slot holding both
    readers' candidates; a character only the second reader saw becomes an optional slot.
    """
    if not secondary:
        return list(primary)
    if not primary:
        return list(secondary)
    result = []
    for i, j in pairs(text(primary), text(secondary)):
        if i is None:
            other = secondary[j]
            if other.c.isalnum():
                result.append(Char("", 0.0, {"": 0.5, **{c: q for c, q in other.alts.items() if c}}))
            continue
        char = primary[i]
        if j is None:
            result.append(char)
            continue
        other = secondary[j]
        if other.c == char.c:
            alts = dict(char.alts)
            for candidate, q in other.alts.items():
                alts[candidate] = max(alts.get(candidate, 0.0), q)
            result.append(Char(char.c, 1 - (1 - char.p) * (1 - other.p), alts, char.agreed + other.agreed))
        else:  # The candidate both readers back most leads the slot (ties: the primary reader's).
            alts = {c: (char.alts.get(c, 0.0) + other.alts.get(c, 0.0)) / 2 for c in {**char.alts, **other.alts}}
            choice = max(alts, key=lambda c: (alts[c], c == char.c))
            result.append(Char(choice, 0.0, alts, 1))
    return result


def agreement(a, b):
    """Similarity of two readings over letters and digits only (0..1)."""
    a, b = [c for c in a if c.isalnum()], [c for c in b if c.isalnum()]
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def segments(chars, threshold=0.95, top_k=4):
    """Locked text runs (str) and slots (tuple of alternatives, the reader's choice first)."""
    result = []
    for char in chars:
        ranked = [c for c, _ in sorted(char.alts.items(), key=lambda item: -item[1]) if c != char.c]
        options = (char.c, *ranked[:top_k - 1])
        if char.p >= threshold or len(options) == 1:
            if char.c:
                if result and isinstance(result[-1], str):
                    result[-1] += char.c
                else:
                    result.append(char.c)
        else:
            result.append(options)
    return result


def slots(segs):
    return sum(isinstance(segment, tuple) for segment in segs)


def best(segs):
    return "".join(segment if isinstance(segment, str) else segment[0] for segment in segs)


def template(segs, escape=str):
    return "".join(escape(segment) if isinstance(segment, str)
                   else OPEN + BAR.join(escape(option) for option in segment) + CLOSE for segment in segs)


def literal(string):
    """Escape for regex syntax shared by Python `re` and xgrammar."""
    return "".join("\\n" if c == "\n" else "\\t" if c == "\t" else "\\" + c if c in _SPECIAL else c
                   for c in string)


def regex(segs, escape=str):
    return "".join(literal(escape(segment)) if isinstance(segment, str)
                   else "(" + "|".join(literal(escape(option)) for option in segment) + ")" for segment in segs)
