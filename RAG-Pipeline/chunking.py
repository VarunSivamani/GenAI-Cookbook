import os
from typing import List, Optional
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
MAX_TOKENS = tokenizer.model_max_length

def chunking(document, tokenizer, max_tokens) -> List:
    """
    📦 Splits a document into manageable chunks using HybridChunker.

    Args:
        document: 📄 The document object to be chunked.
        tokenizer: 🔤 Tokenizer used to tokenize text.
        max_tokens (int): 🔢 Maximum number of tokens allowed per chunk.

    Returns:
        List of chunks 🧩 if successful, otherwise an empty list.
    """

    # 🛡️ Check if a valid document is provided
    if document is None:
        print("❌ [Error] No document provided.")
        return []

    # 🛡️ Validate tokenizer
    if tokenizer is None:
        print("❌ [Error] Tokenizer is missing.")
        return []

    # 🛡️ Validate max_tokens
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        print("⚠️ [Warning] max_tokens should be a positive integer.")
        return []

    try:
        # 🧠 Initialize the HybridChunker
        print("🧠 Initializing HybridChunker...")
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=True,
        )

        # ✂️ Begin chunking
        print("🔧 Chunking in progress...")
        chunk_iter = chunker.chunk(dl_doc=document)
        chunks = list(chunk_iter)

        # ✅ Done
        print(f"✅ Chunking complete. Total chunks created: {len(chunks)}")

        return chunks

    except Exception as e:
        print(f"💥 [Error] Chunking failed: {e}")
        return []
