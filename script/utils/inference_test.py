import gradio as gr
from unsloth import FastModel
from transformers import TextIteratorStreamer
from threading import Thread
from init_prompt import SYSTEM_PROMPT,Original_system_prompt,TEST,NORMAL,CONTINUE_LEARN
import re

# To disable the parallelism warning from tokenizers

model_name = "checkpoint-3062"
# model_name = "model_epoch_1"
# model_name = "unsloth/gemma-3n-E4B-it"

seq_lenth = 4096

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name, 
    max_seq_length = seq_lenth,
    load_in_4bit = True,
    full_finetuning=False,
    device_map = {"": "cuda:0"},
)
model.eval()

print("✅ 模型加载完成！")

# --- Define Gradio interaction function (generator version) ---
def chat_interaction_stream(user_input, history):
    # 1. Convert history format (same as before)
    messages = [{"role": "system", "content":f""}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": f"{user_msg}"})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": f"{user_input}"})

    # 2. Apply chat template and tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        enable_thinking = True,
        add_generation_prompt = True,
    )
    # Gemma3nProcessor expects `text=` as keyword to avoid binding to `images`.
    inputs = tokenizer(text=text, return_tensors="pt").to("cuda")

    # 3. Create TextIteratorStreamer instance
    # skip_prompt=True: Do not include the input prompt in the output
    # skip_special_tokens=True: Do not output special tokens such as <eos>
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 4. Define generation kwargs and pass streamer

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=seq_lenth,

        temperature=0.7,
        top_p=0.8,
        top_k=20,

        # temperature=1.0,
        # top_p=0.95,
        # top_k=64,
        repetition_penalty=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    
    # 5. Create and start a new thread to run model.generate
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 6. Main thread iterates streamer and yields text in real time
    bot_response = ""
    sentence_buffer = ""
    
    for new_text in streamer:
        sentence_buffer += new_text
        
        # Check for complete sentences (ending with Chinese period/question/exclamation or English period)
        sentences = re.split(r'([。！？\.]\s*)', sentence_buffer)
        
        # If there are complete sentences, output them
        if len(sentences) > 1:
            # Handle all complete sentences
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    complete_sentence = sentences[i] + sentences[i + 1]
                    if complete_sentence.strip():
                        bot_response += complete_sentence
                        yield bot_response
            
            # Keep the last incomplete sentence
            sentence_buffer = sentences[-1] if sentences else ""
    
    # Output the remaining text
    if sentence_buffer.strip():
        bot_response += sentence_buffer
        yield bot_response

# --- Launch Gradio UI ---
# Use the same Gradio code; it auto-detects normal vs generator function
chat_ui = gr.ChatInterface(
    fn=chat_interaction_stream, # Note: switched to the new streaming function
    title="与 流萤 对话 💬",
    description="你是开拓者，正在与萨姆驾驶员流萤对话。在下方输入你想说的话。",
    examples=[["你好！你是谁？"], ["我感觉自己干啥都不行"], ["你来自哪里？"]],
    theme="soft",
    chatbot=gr.Chatbot(
        show_copy_button=True,
        bubble_full_width=False,
        show_share_button=False,
    )
)

# Run the web app
chat_ui.launch(share=False)
