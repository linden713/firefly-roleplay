#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, re, unicodedata, functools, hashlib
from pathlib import Path

SYSTEM_TEMPLATE = "你是崩坏星穹铁道的角色{ASSISTANT}，请始终保持角色设定和语气"

# ---------- Constants and Utilities ----------
RE_HTML = re.compile(r'<[^>]*>')
RE_WS = re.compile(r'\s+')
RE_SENT_SPLIT = re.compile(r'(?<=[。！？!?…]|—)|\n+')
RE_PUNCT = re.compile(r"[，,。．\.！!？\?：:；;、~/·\-\—\–\_“”\"'‘’（）\(\)\[\]\{\}<>《》【】]+")

def clean_text(text: str) -> str:
    if not text: return ""
    text = RE_HTML.sub('', text)
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
def load_and_preprocess(input_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    turns = []
    for d in raw:
        s = (d.get("S") or "").strip() or "开拓者"
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
def collapse_to_alternating(window, assistant_speaker):
    seq = []
    for d in window:
        role = "assistant" if d["S"] == assistant_speaker else "user"
        piece = d["T"] if role == "assistant" else f"{d['S']}：{d['T']}"
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
def build_samples_expand_all_speakers(turns, raw_window=12, target_len=6):
    data = []
    n = len(turns)
    if n == 0: return data

    windows = [(0, n)] if n <= raw_window else [(i, i + raw_window) for i in range(0, n - raw_window + 1)]
    global_seen_sent_hash = set()   # Store 64-bit fingerprints of assistant sentences only

    for l, r in windows:
        window = turns[l:r]
        unique_speakers = list(dict.fromkeys([d["S"] for d in window]))

        for sp in unique_speakers:
            seq = collapse_to_alternating(window, sp)
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

            system_msg = SYSTEM_TEMPLATE.format(ASSISTANT=sp)
            messages = [{"role": "system", "content": system_msg}]
            messages.extend(seq)
            data.append({"messages": messages})
    return data

def write_jsonl(items, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
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
    parser = argparse.ArgumentParser(description="多轮对话数据生成（user 开头、assistant 结尾；assistant 句子级全局唯一，高速版）")
    parser.add_argument("--input", "-i", default="/home/lch/firefly-qwen-roleplay/dataset/raw/SR_Talk_CH.json") 
    parser.add_argument("--output", "-o", default="/home/lch/firefly-qwen-roleplay/dataset/multiturn_expand_all.jsonl")
    parser.add_argument("--target-len", "-t", type=int, default=6)
    parser.add_argument("--raw-window", "-w", type=int, default=12)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误：输入文件 {args.input} 不存在")
        return

    turns = load_and_preprocess(args.input)
    merged = merge_consecutive_by_speaker(turns)
    samples = build_samples_expand_all_speakers(
        merged,
        raw_window=max(args.raw_window, args.target_len),
        target_len=args.target_len,
    )
    write_jsonl(samples, args.output)
    print(f"成功生成 {len(samples)} 条样本（assistant 句子级全局唯一｜高速）")
    print(f"数据已保存到: {args.output}")

if __name__ == "__main__":
    main()

