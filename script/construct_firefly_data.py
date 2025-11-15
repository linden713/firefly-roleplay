#!/usr/bin/env python3

import json
import argparse
import re
from pathlib import Path

RE_RUBY_OPEN = re.compile(r"\{RUBY_B#[^}]*\}")
RE_RUBY_CLOSE = re.compile(r"\{RUBY_E#\}")

def clean_text(text):
    """
    Clean HTML tags and other unnecessary markers from text.
    """
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

def extract_firefly_chatml_data(input_file, output_file, context_window=3):
    """
    Extract Firefly's lines from Honkai: Star Rail dialogue JSON
    and build a ChatML-formatted dataset.
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSONL file path
        context_window: Context window size (use previous N utterances as context)
    """
    
    # Read raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    # System message
    system_message = "You are Firefly from Honkai: Star Rail. Always stay in character and speak in her tone and personality. 你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气"
    
    chatml_data = []
    i = 0
    
    # Preprocess: clean text and replace empty speakers
    for dialogue in dialogues:
        # Clean text content
        if "T" in dialogue:
            dialogue["T"] = clean_text(dialogue["T"])
        
        # Replace empty speaker with "Pioneer"
        if dialogue.get("S", "").strip() == "":
            dialogue["S"] = "开拓者"
    
    # Iterate all dialogues and find segments where Firefly speaks
    while i < len(dialogues):
        if dialogues[i].get("S") == "流萤":
            # Collect consecutive Firefly lines
            firefly_texts = []
            firefly_start = i
            
            # Merge consecutive Firefly lines
            while i < len(dialogues) and dialogues[i].get("S") == "流萤":
                text = dialogues[i].get("T", "").strip()
                if text:
                    firefly_texts.append(text)
                i += 1
            
            # If Firefly has content, process this segment
            if firefly_texts:
                # Build context: collect previous lines as user messages
                context_messages = []
                last_speaker = None # Track the last speaker
                
                # Look backward for context utterances
                start_idx = max(0, firefly_start - context_window)
                for j in range(start_idx, firefly_start):
                    prev_dialogue = dialogues[j]
                    speaker = prev_dialogue.get("S", "未知")
                    text = prev_dialogue.get("T", "").strip()
                    
                    # Skip empty text; keep lines not spoken by Firefly
                    if text and speaker != "流萤":
                        # If same speaker as previous, do not add prefix
                        if speaker == last_speaker:
                            context_messages.append(text)
                        else:
                            context_messages.append(f"{speaker}：{text}")
                        last_speaker = speaker
                
                # Build user message (context)
                if context_messages:
                    user_content = "\n".join(context_messages)
                else:
                    user_content = "请开始对话"
                
                # Join Firefly's consecutive lines with newlines
                assistant_content = "\n".join(firefly_texts)
                
                # Build ChatML messages
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}, 
                    {"role": "assistant", "content": assistant_content}
                ]
                
                chatml_data.append({"messages": messages})
        else:
            i += 1
    
    # Write JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in chatml_data:
            f.write(json.dumps(item, ensure_ascii=False, indent=4) + '\n')
    
    print(f"成功提取 {len(chatml_data)} 条流萤对话数据")
    print(f"数据已保存到: {output_file}")
    
    return len(chatml_data)

def main():
    parser = argparse.ArgumentParser(description="提取流萤台词生成ChatML数据集")
    parser.add_argument("--input", "-i", 
                       default="dataset/raw/SR_Talk_CH.json",
                       help="输入JSON文件路径")
    parser.add_argument("--output", "-o",
                       default="dataset/firefly_chatml_story_dataset.jsonl", 
                       help="输出JSONL文件路径")
    parser.add_argument("--context-window", "-c", type=int, default=3,
                       help="上下文窗口大小（默认3）")
    
    args = parser.parse_args()
    
    # Check whether the input file exists
    if not Path(args.input).exists():
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    # Extract data
    extract_firefly_chatml_data(args.input, args.output, args.context_window)

if __name__ == "__main__":
    main()
