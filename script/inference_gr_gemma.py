import gradio as gr
from unsloth import FastModel
from transformers import TextIteratorStreamer
from threading import Thread
from utils.init_prompt import SYSTEM_PROMPT, Original_system_prompt, TEST, NORMAL, NORMAL_EN
from utils.utils import load_params
import os
import torch

# To disable the parallelism warning from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load generation config
gen_cfg = load_params("generation")

# Model configuration
model_name = "unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit"
seq_length = 2048
load_in_4bit = True
full_finetuning = False
checkpoint_path = "outputs/checkpoint-732"

print(f"📚 Loading model: {model_name}...")
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=seq_length,
    load_in_4bit=load_in_4bit,
    full_finetuning=full_finetuning,
)

# Load adapter
print(f"Loading adapter from {checkpoint_path}...")
model.load_adapter(checkpoint_path)

# Enable native 2x faster inference
FastModel.for_inference(model)
print("✅ Model loaded and optimized for inference!")

def chat_interaction_stream(user_input, history, is_chinese=False):
    # 1. Prepare messages
    prompt_map = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "Original_system_prompt": Original_system_prompt,
        "TEST": TEST,
        "NORMAL": NORMAL,
        "NORMAL_EN": NORMAL_EN,
    }
    # Default to NORMAL_EN as per previous script logic (or make it selectable if needed, but keeping simple for now)
    system_prompt = NORMAL_EN 
    
    messages = [{"role": "system", "content": system_prompt}]
    
    user_prefix = "开拓者" if is_chinese else "Trailblazer"
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": f"{user_prefix}: {user_msg}"})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": f"{user_prefix}: {user_input}"})

    # 2. Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text=text, return_tensors="pt").to("cuda")

    # 3. Streamer setup
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 4. Generation parameters (Matching evaluate_gemma3_eval_only.py)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 256)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.8)),
        top_k=int(gen_cfg.get("top_k", 20)),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    
    # 5. Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 6. Yield output
    bot_response = ""
    for new_text in streamer:
        bot_response += new_text
        yield bot_response

# --- Launch Gradio UI ---
chat_ui = gr.ChatInterface(
    fn=chat_interaction_stream,
    title="与 流萤 对话 💬",
    description="你是开拓者，正在与萨姆驾驶员流萤对话。在下方输入你想说的话。",
    examples=[["你好！你是谁？"], ["我感觉自己干啥都不行"], ["你来自哪里？"]],
    theme="soft",
    chatbot=gr.Chatbot(
        show_copy_button=True,
        bubble_full_width=False,
        show_share_button=False,
    ),
    additional_inputs=[
        gr.Checkbox(label="Use Chinese Prefix (开拓者)", value=False)
    ]
)

if __name__ == "__main__":
    chat_ui.launch(share=False)
