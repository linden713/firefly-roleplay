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
        # self.model.save_pretrained(f"model_epoch_{int(state.epoch)}", self.tokenizer)
        print(f"✅ model已保存")


def load_chatml_dataset(file_path):
    """
    Robustly load a file that contains many pretty-printed JSON objects back-to-back,
    each with a top-level {"messages": [...]} structure.

    - Handles multi-line objects (not strict JSONL)
    - Ignores braces inside strings
    - Skips line comments starting with // when outside strings (JSON5-style convenience)
    """
    print(f"📖 正在读取数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        s = f.read()

    conversations = []

    # State machine over the entire file
    buf_chars = []
    depth = 0
    in_str = False
    escape = False
    i = 0
    n = len(s)

    def flush_buffer():
        nonlocal buf_chars
        raw = ''.join(buf_chars).strip()
        buf_chars = []
        if not raw:
            return
        try:
            obj = json.loads(raw)
            msgs = obj.get("messages") if isinstance(obj, dict) else None
            if isinstance(msgs, list):
                conversations.append(msgs)
        except Exception as e:
            # Best effort: skip invalid segments but continue parsing
            print(f"⚠️ 跳过无效JSON对象: {e}")

    while i < n:
        ch = s[i]

        # Handle comment starts when not in string: // ... \n
        if not in_str and ch == '/' and i + 1 < n and s[i + 1] == '/':
            # skip until end of line
            i += 2
            while i < n and s[i] not in ('\n', '\r'):
                i += 1
            # keep the newline for structure separation
            if i < n:
                buf_chars.append('\n')
            i += 1
            continue

        # Manage string and escape state
        if ch == '"':
            if not escape:
                in_str = not in_str
            buf_chars.append(ch)
            escape = False
            i += 1
            continue
        if ch == '\\':
            # Propagate escape inside strings
            buf_chars.append(ch)
            escape = not escape if in_str else False
            i += 1
            continue
        else:
            escape = False

        # Track braces only when not inside string
        if not in_str:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1

        buf_chars.append(ch)

        # When a full JSON object is closed, flush buffer
        if depth == 0 and not in_str and ''.join(buf_chars).strip():
            # Heuristic: only flush if buffer looks like a JSON object (starts with '{')
            # to avoid flushing stray whitespace/newlines between objects.
            tmp = ''.join(buf_chars).lstrip()
            if tmp.startswith('{') and tmp.rstrip().endswith('}'):
                flush_buffer()
        i += 1

    # Flush any trailing object
    if depth == 0:
        flush_buffer()

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
    # Default param yamls are in script/param
    # This file is located at script/utils/utils.py
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, "param"))


def load_params(section: str, yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML parameter configuration from a specific file.

    - section: The name of the configuration section, which corresponds to the YAML filename
               (e.g., "training_continue" loads "training_continue.yaml").
    - yaml_path: The directory where the YAML files are located. Defaults to 'script/param'.
    """
    param_dir = yaml_path or _default_param_path()
    path = os.path.join(param_dir, f"{section}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件: {path}")

    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请先安装: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # The new files have the section name as the root key.
    data = cfg.get(section, {})

    # Wrap into a dot-accessible mapping for convenience while preserving dict behavior
    return _to_dotdict(data)


class _DotDict(dict):
    """A simple dict subclass that supports attribute access recursively.

    - Keeps standard dict methods (e.g., .get, .items) for compatibility
    - Allows nested attribute access: cfg.model.name
    - Lists are traversed and any nested dicts are wrapped
    """

    def __init__(self, mapping=None):
        super().__init__()
        mapping = mapping or {}
        for k, v in mapping.items():
            super().__setitem__(k, self._wrap(v))

    def _wrap(self, value):
        if isinstance(value, dict):
            return _DotDict(value)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        # Allow setting internal/private attributes normally
        if key.startswith('_'):
            return super().__setattr__(key, value)
        # Set as dict item (and wrap if needed)
        super().__setitem__(key, self._wrap(value))

    # Keep __setitem__ to ensure wrapping on item assignment as well
    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))


def _to_dotdict(obj: Any) -> Any:
    """Recursively convert dicts to _DotDict while preserving lists and scalars."""
    if isinstance(obj, dict):
        return _DotDict(obj)
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj
