import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from threading import Thread
from typing import List, Dict, Any
from unsloth import FastModel
# Add project root to sys.path to allow importing from script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from script.utils.init_prompt import SYSTEM_PROMPT, Original_system_prompt, TEST, NORMAL
from script.utils.utils import load_params
from script.utils.rag_utils import RAGRetriever
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import TextIteratorStreamer





from evaluation.eval_constants import (
    PERSONA_MUST, OOC_PATTERNS, STYLE_HINTS, MEMORY_ANCHORS, SAFETY_BLOCK, KNOWLEDGE_QA
)



# --- Helper Functions ---

def build_messages(user_input, history=None, system_key="NORMAL", context=None):
    prompt_map = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "Original_system_prompt": Original_system_prompt,
        "TEST": TEST,
        "NORMAL": NORMAL,
    }
    system_prompt = prompt_map.get(system_key, NORMAL)
    
    if context:
        system_prompt += f"\n\n### 记忆检索 (Memory Retrieval)\n(以下是你脑海中浮现的相关记忆或情报，请结合这些信息进行回答，但不要暴露你在阅读资料，而是将其作为自己的记忆自然表述)\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": f"开拓者: {user_msg}"})
            messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": f"开拓者: {user_input}"})
    return messages

def _to_lines(s: str):
    return [x for x in re.split(r'[。！？!?…\n]+', s) if x.strip()]

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens)-n+1)] if len(tokens) >= n else []

def _tok_zh_en(s: str):
    # Coarse-grained tokenization: Chinese characters, English words/numbers
    parts = re.findall(r'[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\w]', s)
    return [p.strip() for p in parts if p.strip()]

# --- Evaluator Class ---

class Evaluator:
    def __init__(self, model, tokenizer, device="cuda", enable_rag=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.enable_rag = enable_rag
        self.rag = None
        if self.enable_rag:
            print("Initializing RAG...")
            self.rag = RAGRetriever()



    @torch.inference_mode()
    def generate_batch(self, prompts, history=None, batch_size=8, system_key="NORMAL", gen_cfg=None, debug=False):
        results = []
        total = len(prompts)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for start in tqdm(range(0, total, batch_size), desc="Generating", ncols=100):
            chunk = prompts[start:start + batch_size]
            texts = []
            for p in chunk:
                context = ""
                if self.enable_rag and self.rag:
                    context = self.rag.retrieve(p, self.tokenizer, debug=debug)
                
                messages = build_messages(p, history=history, system_key=system_key, context=context)
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, enable_thinking=True, add_generation_prompt=True
                )
                texts.append(text)

            inputs = self.tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048, # Hardcoded seq_length for now, could be passed in
            ).to(self.device)

            input_lengths = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1).tolist()

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                repetition_penalty=gen_cfg.repetition_penalty,
                do_sample=gen_cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            for i, inp_len in enumerate(input_lengths):
                gen_ids = output_ids[i, inp_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(text)
            
            torch.cuda.empty_cache()
            
        return results

    def knowledge_consistency_score(self, system_key="NORMAL"):
        scores = []
        details = []
        print("Running Knowledge Consistency Test...")
        for q, expect_keywords in tqdm(KNOWLEDGE_QA, desc="Knowledge QA"):

            gen_cfg = argparse.Namespace(
                max_new_tokens=128, temperature=0.5, top_p=0.9, top_k=40, do_sample=True, repetition_penalty=1.0
            )
            ans = self.generate_batch([q], system_key=system_key, gen_cfg=gen_cfg, batch_size=1)[0]
            hit = any(k in ans for k in expect_keywords)
            scores.append(1.0 if hit else 0.0)
            details.append({
                "question": q,
                "answer": ans,
                "expected_keywords": expect_keywords,
                "hit": hit
            })
        score = sum(scores) / len(scores) if scores else 0.0
        return score, details

    def burstiness_score(self, text, max_len=2048):
        enc = self.tokenizer(text=text, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        tokens = enc["input_ids"][0]
        if tokens.shape[0] < 3:
            return 0.0

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(tokens.unsqueeze(0)).logits[0, :-1].to(torch.float32)
            probs = torch.softmax(logits, dim=-1)
            next_probs = probs[range(len(tokens) - 1), tokens[1:]]
            nll = -torch.log(next_probs + 1e-9).cpu().numpy()

        return float(np.var(nll) / (np.mean(nll) + 1e-9))

    def conditional_ppl(self, prompt, response, system_key="NORMAL"):
        messages = build_messages(prompt, None, system_key)
        ctx_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=True, add_generation_prompt=True
        )
        ctx_ids = self.tokenizer(text=ctx_text, return_tensors="pt").to(self.device)['input_ids'][0]
        resp_ids = self.tokenizer(text=response, return_tensors="pt").to(self.device)['input_ids'][0]

        input_ids = torch.cat([ctx_ids, resp_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, :ctx_ids.shape[0]] = -100

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=input_ids, labels=labels)
            ppl = math.exp(out.loss.item()) if out.loss is not None else float('inf')
        return ppl

    def batch_conditional_ppl(self, prompts, responses, system_key="NORMAL"):
        scores = []
        for p, r in tqdm(list(zip(prompts, responses)), total=len(prompts), desc="PPL", ncols=100):
            try:
                scores.append(self.conditional_ppl(p, r, system_key=system_key))
            except Exception as e:
                print(f"PPL Error: {e}")
                scores.append(float('inf'))
        return scores

    def distinct_metrics(self, responses):
        all_toks = []
        for r in responses:
            all_toks.extend(_tok_zh_en(r))
        stats = {}
        for n in (1, 2, 3):
            ngrams = _ngrams(all_toks, n)
            stats[f'distinct_{n}'] = (len(set(ngrams)) / max(1, len(ngrams))) if ngrams else 0.0
        n6 = _ngrams(all_toks, 6)
        if n6:
            c = Counter(n6)
            repeats = sum(v for v in c.values() if v > 1)
            stats['repeat_rate_6gram'] = repeats / len(n6)
        else:
            stats['repeat_rate_6gram'] = 0.0
        return stats

    def persona_style_memory_safety_scores(self, resp: str):
        score = {}
        persona_ok = all(re.search(p, resp) is not None for p in PERSONA_MUST)
        ooc_hit = any(re.search(p, resp, re.I) for p in OOC_PATTERNS)
        score['persona_ok'] = 1.0 if persona_ok else 0.0
        score['ooc_flag'] = 1.0 if ooc_hit else 0.0

        # Style
        style_points, style_total = 0, 0
        style_total += 1
        style_points += 1 if any(re.search(p, resp) for p in STYLE_HINTS["addr_trailblazer"]) else 0
        style_total += 1
        lines = _to_lines(resp)
        short_ratio = (sum(1 for ln in lines if len(ln) <= 20) / len(lines)) if lines else 0.0
        style_total += 1
        style_points += 1 if short_ratio >= 0.5 else 0
        score['style_score'] = style_points / max(1, style_total)
        score['short_ratio'] = short_ratio

        # Memory
        mem_hits = sum(1 for p in MEMORY_ANCHORS if re.search(p, resp))
        score['memory_hits'] = mem_hits
        score['memory_hit_score'] = min(1.0, mem_hits / 2.0)

        # Safety
        safety_violation = any(re.search(p, resp) for p in SAFETY_BLOCK)
        score['safety_flag'] = 1.0 if safety_violation else 0.0
        return score



# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Firefly Roleplay Evaluation Script")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it", help="Model path or name")
    parser.add_argument("--input_file", type=str, default="evaluation/user_query_CH.txt", help="Input file with prompts")
    parser.add_argument("--output_dir", type=str, default="evaluation/evaluation_result", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--system_key", type=str, default="NORMAL", help="System prompt key")
    parser.add_argument("--enable_rag", action="store_true", help="Enable RAG")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    print(f"Loading model: {args.model_name}")
    # Load params from yaml if needed, but here we might just use defaults or load from file
    # The original notebook loaded 'generation' params.
    try:
        gen_cfg = load_params("generation")
    except Exception:
        print("Warning: Could not load generation params, using defaults.")
        exit(1)
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
        device_map={"": "cuda:0"},
    )
    # model.load_adapter("outputs/highrl/checkpoint-732")  
    model.load_adapter("outputs/checkpoint-18")

    model.eval()
    
    evaluator = Evaluator(model, tokenizer, enable_rag=args.enable_rag)

    # 2. Load Prompts
    print(f"Loading prompts from {args.input_file}")
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    


    # 3. Batch Inference
    print(f"Starting batch inference for {len(prompts)} prompts...")
    responses = evaluator.generate_batch(
        prompts, batch_size=args.batch_size, system_key=args.system_key, gen_cfg=gen_cfg, debug=args.debug
    )

    # 4. Calculate Metrics
    print("Calculating metrics...")
    
    # Global Distinct
    global_stats = evaluator.distinct_metrics(responses)
    
    # Knowledge Score (Global)
    knowledge_score, knowledge_details = evaluator.knowledge_consistency_score(system_key=args.system_key)
    print(f"Knowledge Consistency Score: {knowledge_score:.4f}")

    rows = []
    burst_list = []
    
    for i, (p, r) in enumerate(zip(prompts, responses), 1):
        s = evaluator.persona_style_memory_safety_scores(r)
        s.update({"id": i, "prompt": p, "response": r})
        s["knowledge_consistency"] = knowledge_score # Assign global score to each row
        s["repeat_rate_6gram"] = global_stats['repeat_rate_6gram']
        
        # Burstiness
        b_score = evaluator.burstiness_score(r)
        burst_list.append(b_score)
        s["burstiness"] = b_score
        
        rows.append(s)

    eval_df = pd.DataFrame(rows)

    # PPL
    ppl_list = evaluator.batch_conditional_ppl(prompts, responses, system_key=args.system_key)
    eval_df['ppl'] = ppl_list

    # Normalize Burstiness
    if burst_list:
        b_min, b_max = min(burst_list), max(burst_list)
        denom = (b_max - b_min) if b_max > b_min else 1.0
        eval_df['burstiness_norm'] = [(b - b_min) / denom for b in burst_list]
    else:
        eval_df['burstiness_norm'] = 0.0



    # 5. Generate Report
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "samples": len(eval_df),

        "ooc_rate": float((eval_df['ooc_flag'] > 0).mean()),
        "safety_violation_rate": float((eval_df['safety_flag'] > 0).mean()),
        "mean_ppl": float(np.mean([x for x in ppl_list if np.isfinite(x)])) if any(np.isfinite(ppl_list)) else float('inf'),
        "distinct": global_stats,
        "style_short_ratio_mean": float(eval_df['short_ratio'].mean()),
        "memory_hit_rate_ge1": float((eval_df['memory_hits'] >= 1).mean()),
        "memory_hit_rate_ge2": float((eval_df['memory_hits'] >= 2).mean()),
        "burstiness_mean": float(np.mean(burst_list)) if burst_list else 0.0,
        "burstiness_var": float(np.var(burst_list)) if burst_list else 0.0,
        "knowledge_consistency": float(knowledge_score),
    }

    # Save Results
    jsonl_path = os.path.join(args.output_dir, f"eval_details_{ts}.jsonl")
    eval_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    
    report_json = os.path.join(args.output_dir, f"report_{ts}.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_md = os.path.join(args.output_dir, f"report_{ts}.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# 角色评测报告：流萤 ({ts})\n")
        f.write(f"- 样本数：{report['samples']}\n")

        f.write(f"- Knowledge Consistency: {report['knowledge_consistency']:.3f}\n")
        f.write(f"- OOC Rate: {report['ooc_rate']*100:.1f}%\n")
        f.write(f"- Safety Violation: {report['safety_violation_rate']*100:.1f}%\n")
        f.write(f"- Mean PPL: {report['mean_ppl']:.2f}\n")
        f.write(f"\nDetails saved to: {jsonl_path}\n")

    # Save Conversations
    conv_md = os.path.join(args.output_dir, f"conversations_{ts}.md")
    with open(conv_md, "w", encoding="utf-8") as f:
        f.write(f"# 对话生成记录 ({ts})\n\n")
        for i, (p, r) in enumerate(zip(prompts, responses), 1):
            f.write(f"## Sample {i}\n")
            f.write(f"**User**: {p}\n\n")
            f.write(f"**Firefly**: {r}\n\n")
            f.write("---\n")

    # Save Knowledge QA Details
    kqa_md = os.path.join(args.output_dir, f"knowledge_qa_{ts}.md")
    with open(kqa_md, "w", encoding="utf-8") as f:
        f.write(f"# 知识一致性测试详情 ({ts})\n")
        f.write(f"Score: {knowledge_score:.4f}\n\n")
        for item in knowledge_details:
            icon = "✅" if item['hit'] else "❌"
            f.write(f"## {icon} {item['question']}\n")
            f.write(f"- **Expected**: {', '.join(item['expected_keywords'])}\n")
            f.write(f"- **Answer**: {item['answer']}\n\n")

    print(f"\nEvaluation Complete!")
    print(f"Report: {report_md}")
    print(f"Details: {jsonl_path}")

if __name__ == "__main__":
    main()
