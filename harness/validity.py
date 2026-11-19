"""
validity.py - Call validity, as defined in the project spec:
"a well-formed call to an existing tool with schema-valid arguments".
Checks: JSON parsed, function exists, arguments is an object, all required
parameters present, no unknown parameters, and basic type match.
"""

PY_TYPES = {
    "string": str,
    "integer": int,
    "float": (int, float),
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "object": dict,
}


def check_call(parsed, schemas_by_name):
    """Return (is_valid, reason)."""
    if parsed is None:
        return False, "malformed JSON"
    if not isinstance(parsed, dict) or "name" not in parsed:
        return False, "missing name"
    name = parsed["name"]
    if name not in schemas_by_name:
        return False, f"unknown tool {name}"
    args = parsed.get("arguments", {})
    if not isinstance(args, dict):
        return False, "arguments not an object"
    params = schemas_by_name[name]["parameters"]
    props = params.get("properties", {})
    for req in params.get("required", []):
        if req not in args:
            return False, f"missing required '{req}'"
    for key, value in args.items():
        if key not in props:
            return False, f"unknown argument '{key}'"
        want = PY_TYPES.get(props[key].get("type"))
        if want is None:
            continue
        if isinstance(value, bool) and want in (int, (int, float)):
            return False, f"'{key}' should be {props[key]['type']}"
        if not isinstance(value, want):
            return False, f"'{key}' should be {props[key]['type']}"
    return True, "ok"
