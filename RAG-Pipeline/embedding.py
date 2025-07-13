from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_id)
EMBED_DIM = len(model.encode("Hello World"))

# 🔍 Define embedding function (batch version)
def embedding(chunks):
    print("🔍 Starting embedding process...")

    # Initialize storage lists for batch insert
    ids = []
    texts = []
    filenames = []
    page_numbers = []
    titles = []
    embeddings = []

    # 🔁 Process chunks with progress bar
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        text = chunk.text  # or chunk.page_content if using LangChain

        # 📝 Extract metadata
        raw_metadata = {
            "filename": chunk.meta.origin.filename,
            "page_numbers": str(sorted({
                prov.page_no for item in chunk.meta.doc_items for prov in item.prov
            })),
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        }

        # 🧹 Normalize None values to empty strings
        metadata = {k: (v if v is not None else "") for k, v in raw_metadata.items()}

        # 🧬 Generate embedding vector
        vector = model.encode(text)  # Replace `model` with your embedding model

        # 📌 Append fields
        ids.append(f"chunk-{i}")
        texts.append(text)
        filenames.append(metadata["filename"])
        page_numbers.append(metadata["page_numbers"])
        titles.append(metadata["title"])
        embeddings.append(vector)

    # 📥 Batch insert to Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"filename": f, "page_numbers": p, "title": t}
            for f, p, t in zip(filenames, page_numbers, titles)
        ]
    )

    print(f"✅ {len(ids)} Chunks embedded and stored!")
    print("🏁 Embedding complete!")
