"""Generated header guards: one spelling rule and a checked, simultaneous binding rewrite."""

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import PurePosixPath

from unbake import merge

_OPEN = re.compile(r"\A\s*#\s*ifndef\s+(\w+)\s*\n\s*#\s*define\s+\1\s*(?=\n|$)")
_PP = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.M)
_WORD = re.compile(r"[A-Za-z_]\w*")


def name(path: str) -> str:
    """The established include-relative generated spelling, independent of host separators."""
    return re.sub(r"\W", "_", path.removeprefix("include/")).upper()


def binding(text: str) -> str | None:
    """A whole-file, empty object-like guard; a partial or ambiguous activation is not a guard."""
    return _binding(merge.mask(text))


def _binding(masked: str) -> str | None:
    found = _OPEN.match(masked)
    if not found:
        return None
    depth = 0
    for directive in _PP.finditer(masked):
        word = re.match(r"\s*#\s*(\w+)", directive[0])
        if not word:
            continue
        if word[1] in {"if", "ifdef", "ifndef"}:
            depth += 1
        elif word[1] in {"else", "elif"} and depth == 1:
            return None
        elif word[1] == "endif":
            depth -= 1
            if depth == 0:
                return found[1] if not masked[directive.end() :].strip() else None
    return None


class Conflict(ValueError):
    """All independently detectable conflicts, with their exact consumer/owner paths."""

    def __init__(self, findings: list[tuple[str, str]]) -> None:
        self.findings = tuple(sorted(set(findings)))
        super().__init__("; ".join(f"{path}: {reason}" for path, reason in self.findings))


def rewrite(texts: Mapping[str, str], selected: Mapping[str, str]) -> dict[str, str]:
    """Validate every selected binding, destination and consumer before any caller writes.

    Review selects headers explicitly; names never infer provenance. A shared guard, occupied
    destination, macro body, numeric use or ordinary identifier refuses the complete batch.
    """
    masks = {path: merge.mask(text) for path, text in texts.items()}
    owners: dict[str, list[str]] = defaultdict(list)
    for path, masked in masks.items():
        if old := _binding(masked):
            owners[old].append(path)
    problems: list[tuple[str, str]] = []
    replacements: dict[str, str] = {}
    destinations: dict[str, list[str]] = defaultdict(list)
    for path, destination in sorted(selected.items()):
        old = _binding(masks.get(path, ""))
        if old is None:
            problems.append((path, "no unique whole-file guard"))
            continue
        if owners[old] != [path]:
            problems.append((path, f"shared guard {old}: {', '.join(sorted(owners[old]))}"))
        definitions = re.findall(r"^[ \t]*#\s*define\s+" + re.escape(old) + r"\b", masks[path], re.M)
        if len(definitions) != 1:
            problems.append((path, f"guard {old} is redefined"))
        new = name(destination)
        if not re.fullmatch(r"[A-Za-z_]\w*", new):
            problems.append((path, f"destination does not generate a C identifier: {new}"))
        if old != new:
            destinations[new].append(path)
            # Shared ownership already refuses; retaining a spelling still sweeps its consumers.
            replacements[old] = new
    for new, paths in destinations.items():
        if len(paths) > 1:
            problems.extend((path, f"guard destination collides: {new}") for path in paths)
    targets = set(destinations)
    names = "|".join(re.escape(n) for n in sorted(set(replacements) | targets))
    outside = re.compile(r"(?<!\w)(?:-D|-U)?(?:" + names + r")(?!\w)") if names else None
    changed: dict[str, str] = {}
    for path, text in texts.items():
        if not replacements:
            break
        if PurePosixPath(path).suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc", ".s"}:
            if outside:
                problems.extend(
                    (path, f"guard consumer outside C/preprocessor bindings: {hit[0]}")
                    for hit in outside.finditer(text)
                )
            continue
        masked = masks[path]
        directives = list(_PP.finditer(masked))
        sites: list[tuple[int, int, str]] = []
        cursor = 0
        for word in _WORD.finditer(masked):
            value = word[0]
            if value in targets:
                problems.append((path, f"guard destination already used: {value}"))
            if value not in replacements:
                continue
            while cursor < len(directives) and directives[cursor].end() <= word.start():
                cursor += 1
            directive = directives[cursor] if cursor < len(directives) else None
            if directive is None or directive.start() > word.start():
                problems.append((path, f"guard {value} used as an ordinary identifier"))
                continue
            head = re.match(r"\s*#\s*(\w+)\s*", directive[0])
            assert head is not None
            operand = directive[0][head.end() :]
            allowed = head[1] in {"ifdef", "ifndef"} and operand.strip() == value
            allowed |= head[1] == "define" and operand.strip() == value and path in owners[value]
            if head[1] in {"if", "elif"}:
                relative = word.start() - directive.start() - head.end()
                allowed |= any(m.start(1) == relative for m in re.finditer(r"\bdefined\s*\(?\s*(\w+)\s*\)?", operand))
            if not allowed:
                problems.append((path, f"ambiguous guard consumer {value} in #{head[1]}"))
            sites.append((word.start(), word.end(), replacements[value]))
        if sites:
            for start, end, new in reversed(sites):
                text = text[:start] + new + text[end:]
            changed[path] = text
    if problems:
        raise Conflict(problems)
    return changed
