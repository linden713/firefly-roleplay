import json
import os
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback
from transformers import TextStreamer

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None
class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_messages, lora=True, gen_config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_messages = eval_messages
        self.lora = lora
        self.gen_config = gen_config or {}
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n🔍 Epoch {state.epoch} 评估输出:")
        
        # Prepare input
        text = self.tokenizer.apply_chat_template(
            self.eval_messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        
        # For Gemma3nProcessor, the first positional arg is `images`.
        # Always pass text via keyword to avoid misbinding.
        inputs = self.tokenizer(text=text, return_tensors="pt").to("cuda")
        
        # Generation parameters (overridable from YAML)
        max_new_tokens = int(self.gen_config.get("max_new_tokens", 256))
        gemma3_temp = float(self.gen_config.get("temperature", 0.7))
        gemma3_top_p = float(self.gen_config.get("top_p", 0.8))
        gemma3_top_k = int(self.gen_config.get("top_k", 20))

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=gemma3_temp,
                top_p=gemma3_top_p,
                top_k=gemma3_top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            )
        print("=" * 50)
    
    def on_save(self, args, state, control, model=None, **kwargs):
        # Always save the merged model instead of the adapter
        # if self.lora:
        #     # TODO
        #     self.model.save_pretrained_merged(f"model_4bit_epoch_{int(state.epoch)}", self.tokenizer, save_method="merged_16bit")
        # else:
        self.model.save_pretrained(f"model_epoch_{int(state.epoch)}", self.tokenizer)
        print(f"✅ model已保存")


def load_chatml_dataset(file_path):
    conversations = []
    print(f"📖 正在读取数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove comment lines
    lines = content.split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith('//')]
    content = '\n'.join(filtered_lines)
    
    # Split multi-line JSON objects
    json_objects = []
    current_obj = ""
    brace_count = 0
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        current_obj += line + '\n'
        brace_count += line.count('{') - line.count('}')
        
        if brace_count == 0 and current_obj.strip():
            try:
                data = json.loads(current_obj.strip())
                if "messages" in data:
                    conversations.append(data["messages"])
                    json_objects.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过无效JSON对象: {e}")
            current_obj = ""
    
    print(f"✅ 数据集读取完成，共加载 {len(conversations)} 条对话")
    return {"conversations": conversations}


def check_conversation_lengths(conversations, tokenizer):
    """
    Compute conversation token lengths (compatible with unsloth + Gemma3nProcessor + chat_template).
    - Do not use datasets.map (avoid multiprocessing/serialization issues)
    - Prefer tokenizer.tokenizer.encode if available
    - Otherwise call Processor.__call__(text=[...]) with keyword argument
    """
    import numpy as np

    print("🔍 正在检查对话长度...")

    # Some environments have Processor(tokenizer=...), where the inner object is a PreTrainedTokenizer(Fast)
    inner_tok = getattr(tokenizer, "tokenizer", None)
    has_encode = hasattr(inner_tok, "encode")

    lengths = []
    apply = tokenizer.apply_chat_template

    for i, convo in enumerate(conversations):
        # 1) Apply chat template (consistent with training)
        text = apply(convo, tokenize=False, add_generation_prompt=False)
        if text is None:
            lengths.append(0); continue
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]
        text = text.strip()
        if not text:
            lengths.append(0); continue

        # 2) Compute tokens
        # if has_encode:
            # Prefer real tokenizer.encode for stability
        token_ids = inner_tok.encode(text, add_special_tokens=False)
            # print("--")
        # else:
        #     # Use Processor; must pass keyword arg text=[...]
        #     out = tokenizer(
        #         text=[text],
        #         add_special_tokens=False,
        #         return_attention_mask=False,
        #         return_token_type_ids=False,
        #     )
        #     token_ids = out["input_ids"][0]

        lengths.append(len(token_ids))



    # 3) Print statistics
    arr = np.array(lengths, dtype=np.int32)
    print("📊 对话长度统计:")
    print(f"   最短: {arr.min()} tokens")
    print(f"   最长: {arr.max()} tokens")
    print(f"   平均: {arr.mean():.1f} tokens")
    print(f"   中位数: {int(np.median(arr))} tokens")
    print(f"   超过2000 tokens的对话: {int((arr > 2000).sum())} 条")
    print(f"   超过1500 tokens的对话: {int((arr > 1500).sum())} 条")
    print(f"   超过1000 tokens的对话: {int((arr > 1000).sum())} 条")
    print(f"   超过512 tokens的对话: {int((arr > 512).sum())} 条")

    return lengths

def formatting_prompts_func(examples,tokenizer):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }


def _default_param_path() -> str:
    # Default param.yaml is at script/param.yaml
    # This file is located at script/utils/utils.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, "param.yaml"))


def load_params(section: Optional[str] = None, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML parameter configuration.

    - section: Return the specified section (e.g. "training_continue", "training_finetune", "inference").
      If None, return the entire configuration.
    - yaml_path: Custom YAML path; by default read script/param.yaml.
    """
    path = yaml_path or _default_param_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件: {path}")
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请先安装: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if section is None:
        return cfg
    return cfg.get(section, {})
