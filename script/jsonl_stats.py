import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List


def try_import_chatml_loader():
    """Try to import the repo's robust ChatML loader, else return None.

    Prefer importing utils.utils from the local script/ path to match training code.
    """
    # 1) Try direct relative import via adding script/ to sys.path
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from utils.utils import load_chatml_dataset  # type: ignore
        return load_chatml_dataset
    except Exception:
        pass
    # 2) Try package-style path if available
    try:
        from script.utils.utils import load_chatml_dataset  # type: ignore
        return load_chatml_dataset
    except Exception:
        return None


def iter_json_objects_chatml(path: str) -> Iterable[Dict[str, Any]]:
    """
    Robust iterator for files containing many pretty-printed JSON objects back-to-back,
    each with a top-level {"messages": [...]} structure.

    - Handles multi-line objects (not strict JSONL)
    - Ignores braces inside strings
    - Skips line comments starting with // when outside strings
    """
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()

    buf_chars: List[str] = []
    depth = 0
    in_str = False
    escape = False
    i = 0
    n = len(s)

    def flush_buffer():
        nonlocal buf_chars
        raw = "".join(buf_chars).strip()
        buf_chars = []
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    while i < n:
        ch = s[i]

        # Skip // comments when outside strings
        if not in_str and ch == "/" and i + 1 < n and s[i + 1] == "/":
            i += 2
            while i < n and s[i] not in ("\n", "\r"):
                i += 1
            if i < n:
                buf_chars.append("\n")
            i += 1
            continue

        # Manage string/escape state
        if ch == '"':
            if not escape:
                in_str = not in_str
            buf_chars.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            buf_chars.append(ch)
            escape = not escape if in_str else False
            i += 1
            continue
        else:
            escape = False

        # Track braces only when not in string
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

        buf_chars.append(ch)

        # When a full JSON object is closed, flush buffer
        if depth == 0 and not in_str and "".join(buf_chars).strip():
            tmp = "".join(buf_chars).lstrip()
            if tmp.startswith("{") and tmp.rstrip().endswith("}"):
                obj = flush_buffer()
                if isinstance(obj, dict):
                    yield obj
        i += 1

    if depth == 0:
        obj = flush_buffer()
        if isinstance(obj, dict):
            yield obj


def extract_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize different common dataset formats into a list of messages.

    Supported patterns:
    - {"messages": [{"role":..., "content": ...}, ...]}
    - {"conversations": [{"from"|"role":..., "value"|"content": ...}, ...]}
    - {"instruction", "input", "output"}
    - {"text": ...}
    Returns list of {role, content} with role in {system,user,assistant,unknown}.
    """
    # ChatML-style
    if isinstance(sample.get("messages"), list):
        out = []
        for m in sample["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "unknown"))
            content = m.get("content", "")
            if isinstance(content, list):
                # If content is segmented (e.g. OpenAI format), join text parts
                parts = []
                for seg in content:
                    if isinstance(seg, dict) and seg.get("type") == "text":
                        parts.append(str(seg.get("text", "")))
                    elif isinstance(seg, str):
                        parts.append(seg)
                content = "\n".join(parts)
            else:
                content = str(content)
            out.append({"role": role, "content": content})
        return out

    # ShareGPT-like
    if isinstance(sample.get("conversations"), list):
        out = []
        for m in sample["conversations"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", m.get("from", "unknown")))
            content = str(m.get("content", m.get("value", "")))
            out.append({"role": role, "content": content})
        return out

    # Alpaca-like
    if any(k in sample for k in ("instruction", "input", "output")):
        sys_msg = sample.get("system")
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        msgs: List[Dict[str, str]] = []
        if sys_msg:
            msgs.append({"role": "system", "content": str(sys_msg)})
        user_text = (inst + ("\n" + inp if inp else "")).strip()
        if user_text:
            msgs.append({"role": "user", "content": user_text})
        if out:
            msgs.append({"role": "assistant", "content": str(out)})
        return msgs

    # Plain text
    if "text" in sample:
        return [{"role": "user", "content": str(sample.get("text", ""))}]

    # Unknown -> flatten string values
    flat = []
    for k, v in sample.items():
        if isinstance(v, str):
            flat.append(f"{k}: {v}")
    return ([{"role": "user", "content": "\n".join(flat)}] if flat else [])


def split_sentences(text: str) -> List[str]:
    # Split by common sentence delimiters, keep non-empty
    parts = re.split(r"[。．！!？?\.\n]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def whitespace_and_cjk_tokenize(text: str) -> List[str]:
    """Approximate tokens without a model: words + individual CJK chars.
    Regex groups words/numbers/underscores as one token, and each CJK char as one.
    """
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)


class Tokenizer:
    def __init__(self):
        self.mode = "fallback"
        self.tok = None
        # Always try Gemma-3 tokenizer locally; fallback if unavailable
        try:
            from transformers import AutoTokenizer  # type: ignore
            self.tok = AutoTokenizer.from_pretrained(
                "unsloth/gemma-3n-E4B-it", local_files_only=True
            )
            self.mode = "hf"
        except Exception:
            self.tok = None
            self.mode = "fallback"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.mode == "hf" and self.tok is not None:
            try:
                return len(self.tok.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return len(whitespace_and_cjk_tokenize(text))


def accumulate_stats(samples: Iterable[Dict[str, Any]], tokenizer: Tokenizer):
    total_samples = 0
    total_messages = 0
    total_messages_excl_system = 0
    total_assistant_msgs = 0
    total_sentences = 0
    total_tokens = 0

    for s in samples:
        msgs = extract_messages(s)
        if not msgs:
            continue
        total_samples += 1

        total_messages += len(msgs)
        ms_excl_sys = [m for m in msgs if m.get("role") != "system"]
        total_messages_excl_system += len(ms_excl_sys)
        total_assistant_msgs += sum(1 for m in msgs if m.get("role") == "assistant")

        for m in msgs:
            content = m.get("content", "") or ""
            total_sentences += len(split_sentences(content))
            total_tokens += tokenizer.count(content)

    avg_messages = (total_messages / total_samples) if total_samples else 0.0
    avg_messages_excl_system = (
        total_messages_excl_system / total_samples if total_samples else 0.0
    )
    # Treat number of assistant messages as number of dialogue rounds (replies)
    avg_turns = (total_assistant_msgs / total_samples) if total_samples else 0.0

    return {
        "total_samples": total_samples,
        "total_messages": total_messages,
        "total_messages_excl_system": total_messages_excl_system,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "avg_messages_per_dialog": round(avg_messages, 3),
        "avg_messages_excl_system_per_dialog": round(avg_messages_excl_system, 3),
        "avg_turns_per_dialog": round(avg_turns, 3),
    }


def load_samples(path: str) -> Iterable[Dict[str, Any]]:
    # If the repo's robust ChatML loader is available and file looks like ChatML,
    # use it to avoid issues with comments/multi-line objects.
    if os.path.isfile(path):
        if path.endswith(".jsonl") or path.endswith(".json"):
            # Use robust local ChatML parser (no heavy deps)
            for obj in iter_json_objects_chatml(path):
                yield obj
        else:
            raise ValueError(f"Unsupported file extension: {path}")
    else:
        # Directory: iterate .jsonl/.json files (non-recursive)
        for name in sorted(os.listdir(path)):
            if not (name.endswith(".jsonl") or name.endswith(".json")):
                continue
            fp = os.path.join(path, name)
            if not os.path.isfile(fp):
                continue
            for obj in load_samples(fp):
                yield obj


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute dataset statistics for JSONL/JSON chat datasets: "
            "samples, sentences, tokens, and avg dialogue turns."
        )
    )
    ap.add_argument("path", help="Path to .jsonl/.json file or a directory")
    args = ap.parse_args()

    tok = Tokenizer()
    stats = accumulate_stats(load_samples(args.path), tok)

    # Pretty print
    print("==== JSONL Stats ====")
    print(f"Total samples            : {stats['total_samples']}")
    print(f"Total messages           : {stats['total_messages']}")
    print(
        f"Total messages (no system): {stats['total_messages_excl_system']}"
    )
    print(f"Total sentences          : {stats['total_sentences']}")
    print(f"Total tokens             : {stats['total_tokens']}")
    print(f"Avg messages/dialog      : {stats['avg_messages_per_dialog']}")
    print(
        f"Avg messages excl sys/dialog: {stats['avg_messages_excl_system_per_dialog']}"
    )
    print(f"Avg turns/dialog (assistant replies): {stats['avg_turns_per_dialog']}")


if __name__ == "__main__":
    main()
