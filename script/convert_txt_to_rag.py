import json
import re
import uuid
import os


MIN_LENGTH = 2

def split_into_sentences(text):
    """
    Splits text into sentences.
    For Chinese (contains Chinese characters): splits by common punctuation, tab, space, and newline.
    For non-Chinese (English): splits by common punctuation and newline.
    """
    # Check if text contains Chinese characters
    is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    
    if is_chinese:
        # Chinese: Split by 。！？.!? and \t and space and \n
        parts = re.split(r'([。！？.!?\t \n])', text)
    else:
        # English: Split by .!? and \n and \t
        parts = re.split(r'([.!?\n\t])', text)
        
    sentences = []
    current_sentence = ""
    
    for part in parts:
        if is_chinese:
            is_delimiter = re.match(r'[。！？.!?\t \n]', part)
        else:
            is_delimiter = re.match(r'[.!?\n\t]', part)
            
        if is_delimiter:
            current_sentence += part
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += part
            
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
        
    return sentences

def convert_txt_to_rag(output_file):
    output_data = []
    
    # 1. Chinese TXT (Line-based Parent)
    path_ch = "/home/lch/firefly-roleplay/dataset/firefly_rag_CH.txt"
    if os.path.exists(path_ch):
        print(f"Processing {path_ch}...")
        with open(path_ch, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.read().split("\n") if l.strip()]
            
        for line in lines:
            parent_content = line
            sentences = split_into_sentences(parent_content)
            
            # Add Parent itself as a "Child" for direct matching? 
            # The plan says "Index Parent (itself)". 
            # In our JSONL format, we have 'child_content' and 'parent_content'.
            # To index the parent itself, we can create an entry where child == parent.
            
            # 1. Parent as Child (Direct Match)
            if len(parent_content) >= MIN_LENGTH:
                output_data.append({
                    "id": str(uuid.uuid4()),
                    "child_content": parent_content,
                    "parent_content": parent_content,
                    "question": "",
                    "metadata": {
                        "source_file": "firefly_rag_CH.txt",
                        "type": "parent_direct"
                    }
                })
            
            # 2. Sentences as Children
            if len(sentences) > 1:
                for sent in sentences:
                    if len(sent) < MIN_LENGTH: continue
                    output_data.append({
                        "id": str(uuid.uuid4()),
                        "child_content": sent,
                        "parent_content": parent_content,
                        "question": "",
                        "metadata": {
                            "source_file": "firefly_rag_CH.txt",
                            "type": "child_sentence"
                        }
                    })

    # 2. English TXT (Paragraph-based Parent)
    path_en = "/home/lch/firefly-roleplay/dataset/firefly_rag_EN.txt"
    if os.path.exists(path_en):
        print(f"Processing {path_en}...")
        with open(path_en, "r", encoding="utf-8") as f:
            paras = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            
        for para in paras:
            parent_content = para
            sentences = split_into_sentences(parent_content)
            
            # 1. Parent as Child (Direct Match)
            if len(parent_content) >= MIN_LENGTH:
                output_data.append({
                    "id": str(uuid.uuid4()),
                    "child_content": parent_content,
                    "parent_content": parent_content,
                    "question": "",
                    "metadata": {
                        "source_file": "firefly_rag_EN.txt",
                        "type": "parent_direct"
                    }
                })
            
            # 2. Sentences as Children
            if len(sentences) > 1:
                for sent in sentences:
                    if len(sent) < MIN_LENGTH: continue
                    output_data.append({
                        "id": str(uuid.uuid4()),
                        "child_content": sent,
                        "parent_content": parent_content,
                        "question": "",
                        "metadata": {
                            "source_file": "firefly_rag_EN.txt",
                            "type": "child_sentence"
                        }
                    })

    print(f"Generated {len(output_data)} entries from TXT files.")
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("Done.")

if __name__ == "__main__":
    output_path = "/home/lch/firefly-roleplay/dataset/firefly_rag_unified.jsonl"
    convert_txt_to_rag(output_path)
