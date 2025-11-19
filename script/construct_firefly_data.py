#!/usr/bin/env python3

import json
import argparse
import re
import unicodedata, functools, hashlib
from pathlib import Path

# --------- Cleaning helpers ---------
RE_RUBY_OPEN = re.compile(r"\{RUBY_B#[^}]*\}")
RE_RUBY_CLOSE = re.compile(r"\{RUBY_E#\}")

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove HTML-style tags, e.g., </color>, <color=xxx>, etc.
    text = re.sub(r'<[^>]*>', '', text)
    # Remove RUBY annotations like: {RUBY_B#纷争之泰坦}尼卡多利{RUBY_E#}
    text = RE_RUBY_OPEN.sub('', text)
    text = RE_RUBY_CLOSE.sub('', text)
    # Normalize and trim extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --------- Turn processing ---------
def _lang_preset(lang: str):
    lang = (lang or "ch").lower()
    if lang in ("en", "english"):
        return {
            "assistant": "Firefly",
            "name_sep": ": ",
            "default_user": "Trailblazer",
            "system_message": "You are Firefly from Honkai: Star Rail. Always stay in character and speak in her tone and personality.",
        }
    # default Chinese
    return {
        "assistant": "流萤",
        "name_sep": "：",
        "default_user": "开拓者",
        "system_message": "你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气",
    }

# ---------- Sentence-level de-dup utils (assistant only) ----------
RE_WS = re.compile(r'\s+')
RE_SENT_SPLIT = re.compile(r'(?<=[。！？!?…]|—)|\n+')
RE_PUNCT = re.compile(r"[，,。．\.！!？\?：:；;、~/·\-\—\–\_“”\"'‘’（）\(\)\[\]\{\}<>《》【】]+")

def sentence_split(text: str):
    parts = RE_SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]

@functools.lru_cache(maxsize=200_000)
def normalize_sent(s: str) -> str:
    t = unicodedata.normalize("NFKC", s).lower()
    t = RE_PUNCT.sub(" ", t)
    t = RE_WS.sub(" ", t).strip()
    return t

def fast_hash64(s: str) -> int:
    h = hashlib.sha1(s.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big', signed=False)
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
    if not turns:
        return []
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

def collapse_to_alternating(window, assistant_speaker: str = "流萤", name_sep: str = "："):
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
    if not seq:
        return None
    # ensure starts with user and ends with assistant
    while seq and seq[0]["role"] != "user":
        seq = seq[1:]
    if not seq:
        return None
    while seq and seq[-1]["role"] != "assistant":
        seq = seq[:-1]
    if not seq:
        return None
    while len(seq) > target_len:
        if len(seq) > 1 and seq[0]["role"] == "user":
            seq = seq[1:]
        else:
            seq = seq[-target_len:]
    if not seq or seq[0]["role"] != "user" or seq[-1]["role"] != "assistant":
        return None
    return seq

def enforce_strict_user_assistant_pairs(seq):
    if not seq:
        return None
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
        if seq[i]["role"] != "user":
            i += 1
            continue
        if i + 1 < n and seq[i+1]["role"] == "assistant":
            paired.append(seq[i])
            paired.append(seq[i+1])
            i += 2
        else:
            i += 1
    if not paired:
        return None
    if paired[-1]["role"] != "assistant":
        return None
    return paired


# --------- Sample generation ---------
def filter_assistant_sentences_no_global_repeat(seq, global_seen_sent_hash):
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
            new_seq.append({"role": "assistant", "content": "\n".join(kept)})
    if not any_assistant:
        return None
    return new_seq

def build_firefly_multi_turn(turns, *, assistant_name: str = "流萤", system_message: str = "", raw_window: int = 12, target_len: int = 6, name_sep: str = "："):
    data = []
    n = len(turns)
    if n == 0:
        return data
    windows = [(0, n)] if n <= raw_window else [(i, i + raw_window) for i in range(0, n - raw_window + 1)]
    global_seen_sent_hash = set()
    for l, r in windows:
        window = turns[l:r]
        seq = collapse_to_alternating(window, assistant_speaker=assistant_name, name_sep=name_sep)
        seq = trim_to_length_keep_ends(seq, target_len=target_len)
        if not seq:
            continue
        seq = filter_assistant_sentences_no_global_repeat(seq, global_seen_sent_hash)
        if not seq:
            continue
        seq = enforce_strict_user_assistant_pairs(seq)
        if not seq:
            continue
        messages = ([{"role": "system", "content": system_message}] if system_message else [])
        messages.extend(seq)
        data.append({"messages": messages})
    return data

def write_jsonl(items, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for it in items:
            obj = {"messages": it.get("messages", [])}
            f.write(json.dumps(obj, ensure_ascii=False, indent=4) + "\n")
    
def merge_samples_by_role(items, max_messages: int = 8):
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


def main():
    parser = argparse.ArgumentParser(description="提取流萤台词并生成多轮ChatML数据集（严格用户开头、助手结尾）")
    parser.add_argument("--input", "-i",
                        default="/home/lch/firefly-roleplay/dataset/raw/SR_Talk_EN.json",
                        help="输入JSON文件路径")
    parser.add_argument("--output", "-o",
                        default="/home/lch/firefly-roleplay/dataset/firefly_chatml_story_dataset_EN.jsonl",
                        help="输出JSONL文件路径")
    parser.add_argument("--lang", choices=["ch", "en"], default="ch", help="输出语言（ch/en）")
    parser.add_argument("--target-len", "-t", type=int, default=6, help="对话长度（不含system）")
    parser.add_argument("--raw-window", "-w", type=int, default=12, help="滑动窗口大小（原始轮次）")
    parser.add_argument("--merge-mode", choices=["none", "role"], default="role", help="样本合并模式：none/role")
    parser.add_argument("--merge-max-messages", type=int, default=10, help="合并后最大消息数（含system）")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"错误：输入文件 {args.input} 不存在")
        return

    cfg = _lang_preset(args.lang)
    system_message = cfg["system_message"]

    turns = load_and_preprocess(args.input, default_user=cfg["default_user"])
    merged = merge_consecutive_by_speaker(turns)
    samples = build_firefly_multi_turn(
        merged,
        assistant_name=cfg["assistant"],
        system_message=system_message,
        raw_window=max(args.raw_window, args.target_len),
        target_len=args.target_len,
        name_sep=cfg["name_sep"],
    )
    if args.merge_mode == "role":
        samples = merge_samples_by_role(samples, max_messages=args.merge_max_messages)
    write_jsonl(samples, args.output)
    print(f"成功生成 {len(samples)} 条多轮流萤对话数据")
    print(f"数据已保存到: {args.output}")


if __name__ == "__main__":
    main()
