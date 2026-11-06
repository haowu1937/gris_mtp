"""HF-style auto-docstring: pull a config's ``Args:`` block onto the consuming object."""

import inspect
import re
from types import UnionType
from typing import Union, get_type_hints

from gensid.config import _CONFIGS


def _unwrap_config(hint):
    """Resolve Optional[X] / Union[X, None] to X if X is a ModelConfig subclass."""
    if isinstance(hint, type) and issubclass(hint, _CONFIGS):
        return hint
    origin = getattr(hint, "__origin__", None)
    if origin in (Union, UnionType):
        for arg in getattr(hint, "__args__", ()):
            found = _unwrap_config(arg)
            if found is not None:
                return found
    return None


def _extract_args_block(docstring: str) -> str:
    """Everything from the ``Args:`` header to the next top-level section."""
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*Args\s*:", line):
            body = [line]
            for next_line in lines[i + 1 :]:
                if re.match(r"^\s*[A-Z][\w ]+:", next_line):
                    break
                body.append(next_line)
            return "\n".join(body)
    return ""


def _splice_at_param(doc: str, param_name: str, block: str, indent: int = 4) -> str:
    """Insert `block` (re-indented) under the Args: entry for `param_name`."""
    lines = doc.splitlines()
    args_idx = next((i for i, line in enumerate(lines) if re.match(r"^\s*Args\s*:", line)), None)
    if args_idx is None:
        return doc
    entry_re = re.compile(rf"^(\s*){re.escape(param_name)}\s*\([^)]*\):")
    entry_idx = next((i for i in range(args_idx + 1, len(lines)) if entry_re.match(lines[i])), None)
    if entry_idx is None:
        return doc
    base = len(lines[entry_idx]) - len(lines[entry_idx].lstrip())
    pad = " " * (base + indent)
    nested = [pad + line if line.strip() else line for line in block.splitlines()]
    end = entry_idx + 1
    while end < len(lines):
        line = lines[end]
        if not line.strip():
            end += 1
            continue
        if len(line) - len(line.lstrip()) <= base:
            break
        end += 1
    return "\n".join(lines[: entry_idx + 1] + nested + lines[end:])


def _expand_config_doc(config_cls, visited=None, depth=0):
    visited = visited or set()
    doc = inspect.getdoc(config_cls) or ""
    if depth >= 3 or config_cls in visited:
        return doc
    visited = visited | {config_cls}
    for fname, ftype in get_type_hints(config_cls).items():
        nested = _unwrap_config(ftype)
        if nested is not None:
            doc = _splice_at_param(doc, fname, _expand_config_doc(nested, visited, depth + 1))
    return doc


def expand_config_docstrings(cls):
    init = cls.__init__
    doc = inspect.getdoc(init) or ""
    for pname, hint in get_type_hints(init).items():
        config_cls = _unwrap_config(hint)
        if config_cls is not None:
            doc = _splice_at_param(doc, pname, _expand_config_doc(config_cls))
    init.__doc__ = doc
    return cls


def auto_docstring(cls=None):
    def deco(cls):
        return expand_config_docstrings(cls)

    return deco(cls) if cls is not None else deco
