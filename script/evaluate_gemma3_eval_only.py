import torch
from unsloth import FastModel
from utils.utils import load_params
from transformers import TextStreamer, TextIteratorStreamer
import gradio as gr
from threading import Thread

# Only read generation from a shared YAML; everything else is inlined below
gen_cfg = load_params("generation")

max_seq_length = 1500

# Define evaluation messages
eval_messages_ch = [
    {"role": "system", "content": "你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气"},
    {"role": "user", "content": "开拓者：流萤，你有什么一直想实现的愿望吗？"},
]
eval_messages_en = [
    {"role": "system", "content": "You are the character Firefly from Honkai: Star Rail. Always stay in character and speak in their tone and personality."},
    {"role": "user", "content": "Trailblazer: Firefly, do you have something you've always wanted to do?"},
]

# Combined messages as in the original script
eval_messages = eval_messages_ch + eval_messages_en

print("📚 正在加载模型和分词器 (checkpoint-732)...")
# Load the model from the specific checkpoint
# model, tokenizer = FastModel.from_pretrained(
#     model_name="/home/lch/firefly-roleplay/outputs/checkpoint-732",
#     max_seq_length=max_seq_length,
#     load_in_4bit=False,
#     load_in_8bit=False,
#     full_finetuning=True,
# )
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)
# Force Gemma-3 chat template for consistency
model.load_adapter("outputs/checkpoint-732")
# tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
# model.eval()

print("✅ 模型和分词器加载完成")

# Enable native 2x faster inference
FastModel.for_inference(model)

print("🚀 开始评估...")
print(f"\n🔍 评估输出:")

# Prepare input
text = tokenizer.apply_chat_template(
    eval_messages,
    tokenize=False,
    enable_thinking=False,
    add_generation_prompt=True,
)

# Tokenize
inputs = tokenizer(text=text, return_tensors="pt").to("cuda")

# Generation parameters (from EvalCallback logic)
max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
gemma3_temp = float(gen_cfg.get("temperature", 0.7))
gemma3_top_p = float(gen_cfg.get("top_p", 0.8))
gemma3_top_k = int(gen_cfg.get("top_k", 20))

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=gemma3_temp,
        top_p=gemma3_top_p,
        top_k=gemma3_top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    )
print("=" * 50)
print("🎊 静态评估完成！")

# --- Gradio Interaction ---
print("🚀 启动 Gradio 界面...")

def chat_stream(user_input, history):
    # Construct messages from history
    messages = []
    # Add a default system prompt
    messages.append({"role": "system", "content": "You are the character Firefly from Honkai: Star Rail. Always stay in character and speak in their tone and personality."})
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": user_input})

    # Prepare input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text=text, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=gemma3_temp,
        top_p=gemma3_top_p,
        top_k=gemma3_top_k,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        streamer=streamer
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    bot_response = ""
    for new_text in streamer:
        bot_response += new_text
        yield bot_response

demo = gr.ChatInterface(
    fn=chat_stream,
    title="Firefly Roleplay Evaluation",
    description="Chat with Firefly (Gemma 3 Checkpoint 732)",
    examples=["Firefly, do you have something you've always wanted to do?", "流萤，你有什么一直想实现的愿望吗？"],
)
demo.launch(share=False)
