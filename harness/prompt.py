"""
prompt.py - Builds Qwen3 chat prompts and parses tool calls.

We render the prompt ourselves (instead of letting the server do it) so that:
  * ALL 30 tool schemas are in the system prompt every turn (baseline rule),
  * sampling is unconstrained (no grammar forced by the server),
  * earlier turns are rendered byte-identically each time (needed later for
    prompt caching to work).
The format follows Qwen3's official chat template, with thinking disabled
(empty <think></think> block), which the template supports via enable_thinking.
"""

import json
import re

SYSTEM_TEXT = (
    "You are a helpful assistant that completes the user's requests by calling "
    "the available tools. Use the tools whenever they are needed, use exact "
    "values from earlier in the conversation, and reply briefly when done."
)

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array"}


def to_json_schema(p):
    """Convert BFCL's parameter types (dict, float, any) to JSON-schema names."""
    p = dict(p)
    t = p.get("type")
    if t == "any":
        p.pop("type")
    elif t in TYPE_MAP:
        p["type"] = TYPE_MAP[t]
    if "properties" in p:
        p["properties"] = {k: to_json_schema(v) for k, v in p["properties"].items()}
    if "items" in p and isinstance(p["items"], dict):
        p["items"] = to_json_schema(p["items"])
    return p


def tool_block(schemas):
    lines = []
    for s in schemas:
        fn = {"name": s["name"], "description": s["description"],
              "parameters": to_json_schema(s["parameters"])}
        lines.append(json.dumps({"type": "function", "function": fn}, ensure_ascii=False))
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + "\n".join(lines) + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def render(messages, schemas):
    """messages: list of {"role": user|assistant|tool, "content": str}"""
    out = [f"<|im_start|>system\n{SYSTEM_TEXT}\n\n{tool_block(schemas)}<|im_end|>\n"]
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "tool":
            parts = []
            while i < len(messages) and messages[i]["role"] == "tool":
                parts.append(f"<tool_response>\n{messages[i]['content']}\n</tool_response>")
                i += 1
            out.append("<|im_start|>user\n" + "\n".join(parts) + "<|im_end|>\n")
            continue
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        i += 1
    out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)


CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text):
    """Return list of (raw_json_text, parsed_or_None). Unparseable calls are kept
    so the validity checker can count them as invalid."""
    calls = []
    for raw in CALL_RE.findall(text):
        try:
            calls.append((raw, json.loads(raw)))
        except json.JSONDecodeError:
            calls.append((raw, None))
    return calls
