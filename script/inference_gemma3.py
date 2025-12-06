import torch
from unsloth import FastModel
from utils.utils import load_params
from utils.init_prompt import NORMAL, OPENING_SCENE, SYSTEM_PROMPT
from transformers import TextStreamer, TextIteratorStreamer
import gradio as gr
from threading import Thread
import torch
import os
from utils.rag_utils import RAGRetriever

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
eval_messages = eval_messages_ch #+ eval_messages_en

print("📚 正在加载模型和分词器 (checkpoint-732)...")
# Load the model from the specific checkpoint

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-e4b-it-unsloth-bnb-4bit",
    # model_name="gemma-3N-continue-learning-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)
# Force Gemma-3 chat template for consistency
# model.load_adapter("outputs/highrl/checkpoint-732")
model.load_adapter("outputs/checkpoint-18")

model.eval()

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
gemma3_repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.0))

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=gemma3_temp,
        top_p=gemma3_top_p,
        top_k=gemma3_top_k,
        repetition_penalty=gemma3_repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    )
print("=" * 50)
print("🎊 静态评估完成！")

# --- Gradio Interaction ---
print("🚀 启动 Gradio 界面...")


# --- RAG Setup ---
rag_retriever = RAGRetriever()

def chat_stream(user_input, history):
    # Construct messages from history
    messages = []
    # Retrieve context
    context = rag_retriever.retrieve(user_input, tokenizer)
    print(f"🔍 Retrieved Context for '{user_input}':\n{context[:200]}..." if context else "No context found.")

    # Add a default system prompt
    base_system_prompt = SYSTEM_PROMPT
    
    if context:
        system_content = f"{base_system_prompt}\n\nRelevant Information (use this to answer if relevant):\n{context}"
    else:
        system_content = base_system_prompt

    messages.append({"role": "system", "content": system_content})
    
    for user_msg, bot_msg in history:
        if user_msg is None:
             messages.append({"role": "user", "content": "（前往约定地点）"})
             messages.append({"role": "assistant", "content": bot_msg})
             continue
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
        repetition_penalty=gemma3_repetition_penalty,
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

def simple_chat_stream(user_input, history):
    # Construct messages from history
    messages = []
    
    # Add a default system prompt
    system_content = SYSTEM_PROMPT
    messages.append({"role": "system", "content": system_content})
    
    for user_msg, bot_msg in history:
        if user_msg is None:
             messages.append({"role": "user", "content": "（前往约定地点）"})
             messages.append({"role": "assistant", "content": bot_msg})
             continue
        messages.append({"role": "user", "content": f"开拓者：{user_msg}"})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": f"开拓者：{user_input}"})

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
        repetition_penalty=gemma3_repetition_penalty,
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

chatbot = gr.Chatbot(value=[(None, OPENING_SCENE)], height=600)

demo = gr.ChatInterface(
    fn=simple_chat_stream,
    chatbot=chatbot,
    title="Firefly Roleplay Evaluation (Simple)",
    description="Chat with Firefly (Gemma 3 Checkpoint 732) - No RAG, No System Prompt",
    examples=["Firefly, do you have something you've always wanted to do?", "流萤，你有什么一直想实现的愿望吗？"],
)
demo.launch(share=False)
