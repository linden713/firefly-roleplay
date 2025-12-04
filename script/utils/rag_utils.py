import torch
import os
import re
from sentence_transformers import SentenceTransformer, util

class RAGRetriever:
    def __init__(self, file_paths=None, embedder_name='paraphrase-multilingual-MiniLM-L12-v2'):
        if file_paths is None:
            self.file_paths = [
                "/home/lch/firefly-roleplay/dataset/firefly_rag_CH.txt",
                "/home/lch/firefly-roleplay/dataset/firefly_rag_EN.txt",
            ]
        else:
            self.file_paths = file_paths
            
        print("⏳ Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
        self.embedder = SentenceTransformer(embedder_name)
        
        self.raw_lines = self.load_rag_data()
        self.search_corpus = []
        self.corpus_embeddings = None
        
        if self.raw_lines:
            self.search_corpus = self.build_search_corpus(self.raw_lines)
            print(f"📚 Loaded {len(self.raw_lines)} raw RAG lines.")
            print(f"🔍 Built {len(self.search_corpus)} search chunks (Lines + Sentences).")
            
            print("⏳ Encoding corpus...")
            corpus_texts = [item["text"] for item in self.search_corpus]
            self.corpus_embeddings = self.embedder.encode(corpus_texts, convert_to_tensor=True)
            print("✅ Corpus encoded.")
        else:
            print("⚠️ No RAG data found. RAG will be disabled.")

    def split_into_sentences(self, text):
        # Split by common sentence terminators (Chinese and English)
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        return [p.strip() for p in parts if p.strip()]

    def load_rag_data(self):
        raw_lines = []
        for path in self.file_paths:
            if os.path.exists(path):
                print(f"Loading RAG data from: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by single newline
                    lines = [line.strip() for line in content.split("\n") if line.strip()]
                    raw_lines.extend(lines)
            else:
                print(f"Warning: RAG file not found: {path}")
        return raw_lines

    def build_search_corpus(self, raw_lines):
        corpus = []
        
        # Method 1: Single lines (Paragraphs)
        for idx, line in enumerate(raw_lines):
            corpus.append({
                "text": line,
                "type": "line",
                "index": idx,
                "source_len": len(raw_lines)
            })
            
        # Method 2: Individual Sentences
        for idx, line in enumerate(raw_lines):
            sentences = self.split_into_sentences(line)
            # If there's only 1 sentence, it's already covered by Method 1 (Line/Paragraph)
            if len(sentences) <= 1:
                continue
                
            for sent in sentences:
                corpus.append({
                    "text": sent,
                    "type": "sentence",
                    "index": idx, # Points to the parent line/paragraph
                    "source_len": len(raw_lines)
                })
            
        return corpus

    def retrieve(self, query, tokenizer, top_k=3, debug=False):
        if self.corpus_embeddings is None or not self.search_corpus:
            return ""
        
        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.search_corpus)))
            
            final_context_blocks = []
            debug_output = []
            TOKEN_THRESHOLD = 20  # Approx 20 tokens
            
            # Sort by score descending
            sorted_results = sorted(zip(top_results.values, top_results.indices), key=lambda x: x[0], reverse=True)
            
            for score, idx in sorted_results:
                if score < 0.3: continue
                
                idx = int(idx)
                result_item = self.search_corpus[idx]
                
                display_content = result_item["text"]
                chunk_type = result_item["type"]
                original_idx = result_item["index"]
                
                # Calculate token count
                try:
                    # tokenizer(text=...) returns a dict with 'input_ids'
                    tokenized = tokenizer(text=display_content, return_tensors="pt")
                    token_count = tokenized.input_ids.shape[1]
                except Exception:
                    # Fallback if something goes wrong
                    token_count = len(display_content) // 2 
                
                # Dynamic Context Expansion for short sentence chunks
                clean_parts = []
                debug_parts = []
                
                if chunk_type == "sentence" and token_count < TOKEN_THRESHOLD:
                    # Add previous if exists
                    if original_idx > 0:
                        prev_text = self.raw_lines[original_idx-1]
                        clean_parts.append(prev_text)
                        debug_parts.append(f"[Prev] {prev_text}")
                    
                    clean_block = display_content # Just the content for now if we are appending
                    # Wait, the logic above was slightly different. Let's stick to the original logic but just fix indentation/flow if needed.
                    # Re-reading original logic:
                    # clean_parts.append(prev_text)
                    # clean_parts.append(display_content)
                    
                    clean_parts.append(display_content)
                    debug_parts.append(f"[Target] {display_content}")
                    
                    # Add next if exists
                    if original_idx < len(self.raw_lines) - 1:
                        next_text = self.raw_lines[original_idx+1]
                        clean_parts.append(next_text)
                        debug_parts.append(f"[Next] {next_text}")
                else:
                    clean_parts.append(display_content)
                    debug_parts.append(display_content)
                
                clean_block = "\n".join(clean_parts)
                debug_block = "\n".join(debug_parts)
                
                # Add to lists
                final_context_blocks.append(clean_block)
                debug_output.append(f"[Score: {score:.4f}]\n{debug_block}")
                
            # Print debug info to terminal
            if debug and debug_output:
                print(f"\n🔍 Debug Retrieval for '{query}':")
                print("\n---\n".join(debug_output))
                print("-" * 50)
                
            return "\n\n---\n\n".join(final_context_blocks)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return ""
