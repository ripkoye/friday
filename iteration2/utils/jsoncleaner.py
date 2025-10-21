import json, re, ast
from typing import Any, Tuple, Optional

# --- regexes
_FENCE_ANY   = re.compile(r"```(\w+)?\s*([\s\S]*?)\s*```", re.M)
_FENCE_JSON  = re.compile(r"```json\s*([\s\S]*?)\s*```", re.I | re.M)
_CODE_FENCE_EDGE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.S)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        return _CODE_FENCE_EDGE.sub("", s).strip()
    return s

def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))

def _strip_comments(s: str) -> str:
    s = re.sub(r"(?m)//.*?$", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return s

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",\s*(?=[}\]])", "", s)

def _balanced_json_slice(s: str) -> Optional[str]:
    # Find first balanced {...} OR [...] (respecting strings)
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        if i < 0:
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < len(s):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return s[i:j+1]
            j += 1
    return None

def _peel_json_strings(obj: Any) -> Any:
    # If json.loads gives a string that itself is JSON, peel layers
    while isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            break
    return obj

def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return _peel_json_strings(json.loads(s))
    except Exception:
        return None

def _try_python_literal(s: str) -> Optional[Any]:
    try:
        obj = ast.literal_eval(s)
        return json.loads(json.dumps(obj))  # normalize
    except Exception:
        return None

def _quote_unquoted_keys(s: str) -> str:
    # Only keys after { or , ; avoid touching values
    pat = re.compile(r'([{\s,])\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:(?=\s)')
    return pat.sub(r'\1"\2":', s)

def extract_json_block(s: str, *, max_scan_len: int = 2_000_000) -> Tuple[Any, str]:
    """
    Robustly extract JSON from LLM output.
    Returns (parsed_object, raw_json_slice).
    Raises ValueError on failure.
    """
    if not isinstance(s, str):
        raise TypeError("extract_json_block expects a string")
    if len(s) > max_scan_len:
        s = s[:max_scan_len]  # safety

    original = s
    s = _normalize_quotes(s).strip()

    # 1) Prefer the first ```json fenced block if present
    m_json = _FENCE_JSON.search(s)
    if m_json:
        candidate = m_json.group(1).strip()
        # fast attempts on json fence
        for attempt in (
            candidate,
            _strip_trailing_commas(_strip_comments(candidate)),
            _strip_code_fences(candidate),
        ):
            obj = _try_json_loads(attempt) or _try_python_literal(attempt)
            if obj is not None:
                return obj, candidate

    # 2) If not, remove generic fence if the whole string is fenced
    if s.startswith("```"):
        s = _strip_code_fences(s)

    # 3) Direct JSON (object/array or double-encoded)
    obj = _try_json_loads(s)
    if obj is not None:
        # Determine a reasonable raw slice for logging
        raw = s if isinstance(obj, (dict, list)) else json.dumps(obj)
        return obj, raw

    # 4) If fully quoted, try inner content
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1]
        obj = _try_json_loads(inner) or _try_python_literal(inner)
        if obj is not None:
            return obj, inner

    # 5) Extract the first balanced JSON-ish slice ({...} or [...])
    candidate = _balanced_json_slice(s)
    if candidate is None:
        raise ValueError("No JSON object/array found in model output")

    cleaned = _strip_trailing_commas(_strip_comments(candidate)).strip()

    # 6) Try clean candidate
    obj = _try_json_loads(cleaned)
    if obj is not None:
        return obj, candidate

    # 7) Try Python literal
    obj = _try_python_literal(cleaned)
    if obj is not None:
        return obj, candidate

    # 8) Last resort: quote unquoted keys, but never alter numeric values
    fixed = _quote_unquoted_keys(cleaned)
    fixed = _strip_trailing_commas(fixed)
    obj = _try_json_loads(fixed) or _try_python_literal(fixed)
    if obj is not None:
        return obj, candidate

    snippet = cleaned[:200].replace("\n", "\\n")
    raise ValueError(f"Could not parse JSON. Snippet: {snippet}")
