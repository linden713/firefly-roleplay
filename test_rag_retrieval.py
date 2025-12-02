from sentence_transformers import SentenceTransformer, util
import torch
import os
import sys

def load_rag_data():
    chunks = []
    file_paths = [
        "/home/lch/firefly-roleplay/dataset/firefly_rag_CH.txt",
        "/home/lch/firefly-roleplay/dataset/firefly_rag_EN.txt"
    ]
    for path in file_paths:
        if os.path.exists(path):
            print(f"Loading RAG data from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split by single newline
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                chunks.extend(lines)
        else:
            print(f"Warning: RAG file not found: {path}")
    return chunks

def main():
    rag_chunks = load_rag_data()
    print(f"📚 Loaded {len(rag_chunks)} RAG chunks.")

    if not rag_chunks:
        print("❌ No data loaded. Exiting.")
        return

    # Initialize Sentence Transformer
    print("⏳ Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("⏳ Encoding corpus...")
    corpus_embeddings = embedder.encode(rag_chunks, convert_to_tensor=True)
    print("✅ Ready! Type your query below (or 'exit' to quit).")

    window_size = 1

    while True:
        try:
            query = input("\n📝 Enter query: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue

            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            
            # Get top 3
            top_results = torch.topk(cos_scores, k=min(3, len(rag_chunks)))
            
            print(f"\n🔍 Top 3 Results for '{query}':")
            print("-" * 50)
            for score, idx in zip(top_results.values, top_results.indices):
                idx = int(idx)
                start_idx = max(0, idx - window_size)
                end_idx = min(len(rag_chunks), idx + window_size + 1)
                
                context_block = "\n".join(rag_chunks[start_idx:end_idx])
                
                print(f"Score: {score:.4f}")
                print(f"Content:\n{context_block}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
