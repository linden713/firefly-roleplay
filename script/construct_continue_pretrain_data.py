#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, re, unicodedata, functools, hashlib
from pathlib import Path

# Language presets
def _lang_preset(lang: str):
    """
    Return language-specific settings:
    - system_template: template string with {ASSISTANT}
    - name_sep: separator between speaker and content in user messages
    - default_user: fallback speaker name when S is empty
    """
    lang = (lang or "zh").lower()
    if lang in ("en", "english"):
        return {
            "system_template": "You are the character {ASSISTANT} from Honkai: Star Rail. Always stay in character and speak in their tone and personality.",
            "name_sep": ": ",
            "default_user": "Trailblazer",
        }
    # default: zh
    return {
        "system_template": "你是崩坏星穹铁道的角色{ASSISTANT}，请始终保持角色设定和语气",
        "name_sep": "：",
        "default_user": "开拓者",
    }

# ---------- Constants and Utilities ----------
RE_HTML = re.compile(r'<[^>]*>')
RE_RUBY_OPEN = re.compile(r"\{RUBY_B#[^}]*\}")
RE_RUBY_CLOSE = re.compile(r"\{RUBY_E#\}")
RE_WS = re.compile(r'\s+')
RE_SENT_SPLIT = re.compile(r'(?<=[。！？!?…]|—)|\n+')
RE_PUNCT = re.compile(r"[，,。．\.！!？\?：:；;、~/·\-\—\–\_“”\"'‘’（）\(\)\[\]\{\}<>《》【】]+")

def clean_text(text: str) -> str:
    if not text: return ""
    text = RE_HTML.sub('', text)
    # Remove RUBY annotations like: {RUBY_B#}xxxx{RUBY_E#}
    text = RE_RUBY_OPEN.sub('', text)
    text = RE_RUBY_CLOSE.sub('', text)
    text = RE_WS.sub(' ', text).strip()
    return text

def sentence_split(text: str):
    parts = RE_SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]

@functools.lru_cache(maxsize=200_000)
def normalize_sent(s: str) -> str:
    # Sentence-level normalization (for global-uniqueness checking)
    t = unicodedata.normalize("NFKC", s).lower()
    t = RE_PUNCT.sub(" ", t)
    t = RE_WS.sub(" ", t).strip()
    return t

def fast_hash64(s: str) -> int:
    # 64-bit fingerprint: take the first 8 bytes of sha1
    h = hashlib.sha1(s.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big', signed=False)

# ---------- Load and Merge ----------
def load_and_preprocess(input_file: str, default_user: str = "开拓者"):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    turns = []
    for d in raw:
        s = (d.get("S") or "").strip() or default_user
        t = clean_text(d.get("T", ""))
        if t:
            turns.append({"S": s, "T": t})
    return turns

def merge_consecutive_by_speaker(turns):
    if not turns: return []
    merged = []
    cur_s, cur_t = turns[0]["S"], [turns[0]["T"]]
    for d in turns[1:]:
        if d["S"] == cur_s:
            cur_t.append(d["T"])
        else:
            merged.append({"S": cur_s, "T": "\n".join(cur_t)})
            cur_s, cur_t = d["S"], [d["T"]]
    merged.append({"S": cur_s, "T": "\n".join(cur_t)})
    return merged

# ---------- Build Sequences ----------
def collapse_to_alternating(window, assistant_speaker, name_sep: str = "："):
    seq = []
    for d in window:
        role = "assistant" if d["S"] == assistant_speaker else "user"
        piece = d["T"] if role == "assistant" else f"{d['S']}{name_sep}{d['T']}"
        if not seq or seq[-1]["role"] != role:
            seq.append({"role": role, "content": piece})
        else:
            seq[-1]["content"] += "\n" + piece
    return seq

def trim_to_length_keep_ends(seq, target_len=6):
    if not seq: return None
    while seq and seq[0]["role"] != "user":
        seq = seq[1:]
    if not seq: return None
    while seq and seq[-1]["role"] != "assistant":
        seq = seq[:-1]
    if not seq: return None
    while len(seq) > target_len:
        if len(seq) > 1 and seq[0]["role"] == "user":
            seq = seq[1:]
        else:
            seq = seq[-target_len:]
    if not seq or seq[0]["role"] != "user" or seq[-1]["role"] != "assistant":
        return None
    return seq

# ---------- Key: assistant sentence-level global uniqueness (fast) ----------
def filter_assistant_sentences_no_global_repeat(seq, global_seen_sent_hash):
    """
    For all assistant segments in `seq`:
      - Split into sentences
      - Normalize + compute 64-bit fingerprint
      - Filter out sentences already seen
    If an assistant segment becomes empty after filtering, drop it.
    If there is no assistant segment left, return None (discard sample).
    """
    new_seq = []
    any_assistant = False

    for m in seq:
        if m["role"] != "assistant":
            new_seq.append(m)
            continue

        sents = sentence_split(m["content"])
        kept = []
        for s in sents:
            norm = normalize_sent(s)
            if not norm: 
                continue
            h = fast_hash64(norm)
            if h in global_seen_sent_hash:
                continue
            kept.append(s)
            global_seen_sent_hash.add(h)

        if kept:
            any_assistant = True
            # Merge back into text (use newlines to avoid concatenation)
            m2 = {"role": "assistant", "content": "\n".join(kept)}
            new_seq.append(m2)
        # If assistant segment fully filtered, skip it (i.e., remove)

    if not any_assistant:
        return None
    return new_seq

# ---------- Sample Generation ----------
def build_samples_expand_all_speakers(turns, raw_window=12, target_len=6, *, system_template: str = "", name_sep: str = "："):
    data = []
    n = len(turns)
    if n == 0: return data

    windows = [(0, n)] if n <= raw_window else [(i, i + raw_window) for i in range(0, n - raw_window + 1)]
    global_seen_sent_hash = set()   # Store 64-bit fingerprints of assistant sentences only

    for l, r in windows:
        window = turns[l:r]
        unique_speakers = list(dict.fromkeys([d["S"] for d in window]))

        for sp in unique_speakers:
            seq = collapse_to_alternating(window, sp, name_sep=name_sep)
            seq = trim_to_length_keep_ends(seq, target_len=target_len)
            if not seq: 
                continue

            # Sentence-level global-unique filtering (fast)
            seq = filter_assistant_sentences_no_global_repeat(seq, global_seen_sent_hash)
            if not seq:
                continue

            seq = enforce_strict_user_assistant_pairs(seq)
            if not seq:
                continue

            system_msg = system_template.format(ASSISTANT=sp)
            messages = [{"role": "system", "content": system_msg}]
            messages.extend(seq)
            # Do not persist meta in final output; only messages are needed downstream
            data.append({"messages": messages})
    return data

# merge_adjacent_samples removed as per user request

def merge_samples_by_role(items, max_messages: int = 8):
    """
    Merge samples that share the same system content (assistant role),
    regardless of adjacency. Preserves overall order of appearance per role.
    """
    if not items:
        return items
    out = []
    role_index = {}

    def sys_text(sample):
        msgs = sample.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            return msgs[0].get("content", "")
        return None

    for sample in items:
        key = sys_text(sample)
        if key is None:
            out.append(sample)
            continue
        if key not in role_index:
            role_index[key] = len(out)
            out.append(sample)
            continue
        tgt = out[role_index[key]]
        merged_len = len(tgt["messages"]) + max(0, len(sample.get("messages", [])) - 1)
        if merged_len <= max_messages:
            tgt["messages"].extend(sample["messages"][1:])
        else:
            role_index[key] = len(out)
            out.append(sample)
    return out

# merge_samples_by_window removed (unused) for simplicity

def write_jsonl(items, output_file: str):
    """
    Write items as pretty-printed JSON objects, one after another,
    each separated by a newline. This matches the multi-line style
    used by our loaders (brace-balanced parsing), e.g.:
    {
        "messages": [
            {"role": "system", ...},
            {"role": "user", ...},
            {"role": "assistant", ...}
        ]
    }
    {
        ...
    }
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for it in items:
            # Ensure only messages are written (strip any accidental metadata)
            obj = {"messages": it.get("messages", [])}
            f.write(json.dumps(obj, ensure_ascii=False, indent=4) + '\n')
def enforce_strict_user_assistant_pairs(seq):
    """
    Normalize sequence to strictly alternating (user, assistant) pairs:
      - Starts with user and ends with assistant
      - Remove any user without a following assistant
      - Remove extra users that cause user→user
      - Return None if constraints not met
    """
    if not seq:
        return None

    # Trim head until a user; trim tail until ending with assistant
    while seq and seq[0]["role"] != "user":
        seq = seq[1:]
    while seq and seq[-1]["role"] != "assistant":
        seq = seq[:-1]
    if not seq:
        return None

    paired = []
    i = 0
    n = len(seq)
    while i < n:
        # Expect user
        if seq[i]["role"] != "user":
            i += 1
            continue
        # Must be immediately followed by assistant; otherwise drop this user
        if i + 1 < n and seq[i+1]["role"] == "assistant":
            paired.append(seq[i])
            paired.append(seq[i+1])
            i += 2
        else:
            i += 1  # Drop unpaired user

    if not paired:
        return None
    # Must still end with assistant
    if paired[-1]["role"] != "assistant":
        return None
    return paired

def main():
    parser = argparse.ArgumentParser(description="Multi-turn dialogue data generation (starts with user, ends with assistant; assistant messages globally unique by sentence, high-speed version)")
    parser.add_argument("--input", "-i", default="dataset/raw/SR_Talk_CH.json") 
    parser.add_argument("--output", "-o", default="dataset/continue_pertrian_CH.jsonl")
    parser.add_argument("--target-len", "-t", type=int, default=6)
    parser.add_argument("--raw-window", "-w", type=int, default=12)
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="Output language affects system prompts, punctuation, and default speaker names")
    parser.add_argument("--merge-mode", choices=["none", "role"], default="none",
                        help="Sample merge mode: none (no merge), role (same role but non-adjacent)")
    parser.add_argument("--merge-max-messages", type=int, default=10, help="Maximum number of messages after merging (including system)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file {args.input} does not exist")
        return


    cfg = _lang_preset(args.lang)
    # Use language preset system template
    sys_tmpl = cfg["system_template"]
    turns = load_and_preprocess(args.input, default_user=cfg["default_user"])
    merged = merge_consecutive_by_speaker(turns)
    samples = build_samples_expand_all_speakers(
        merged,
        raw_window=max(args.raw_window, args.target_len),
        target_len=args.target_len,
        system_template=sys_tmpl,
        name_sep=cfg["name_sep"],
    )
    merge_mode = args.merge_mode
    if merge_mode == "role":
        samples = merge_samples_by_role(samples, max_messages=args.merge_max_messages)
    write_jsonl(samples, args.output)
    print(f"Successfully generated {len(samples)} samples (assistant messages globally unique by sentence | high-speed)")
    print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    main()
