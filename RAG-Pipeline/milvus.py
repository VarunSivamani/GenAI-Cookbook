from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config import SCHEMA_CONFIG

class RAGVectorDB:
    def __init__(self, embed_dim: int):
        """
        Stores connection + collection handle.
        """
        self.embed_dim = embed_dim
        self.collection = None

    def connect(self, uri: str, token: str):
        """
        Connect to Milvus/Zilliz Cloud.
        """
        print("🔗 Connecting to Milvus/Zilliz...")
        connections.connect(
            alias="default",
            uri=uri,
            token=token,
            secure=True
        )
        print("✅ Connected!")
        print("📚 Collections:", utility.list_collections())

    def create_collection(self, index_params: dict = None):
        """
        Create or load collection from SCHEMA_CONFIG.
        """
        fields = []
        for field_def in SCHEMA_CONFIG["fields"]:
            dtype_str = field_def["dtype"]

            if dtype_str == "VARCHAR":
                field = FieldSchema(
                    name=field_def["name"],
                    dtype=DataType.VARCHAR,
                    is_primary=field_def.get("is_primary", False),
                    max_length=field_def["max_length"]
                )
            elif dtype_str == "FLOAT_VECTOR":
                dim = field_def["dim"]
                if dim == "dynamic":
                    dim = self.embed_dim
                field = FieldSchema(
                    name=field_def["name"],
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim
                )
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")

            fields.append(field)

        schema = CollectionSchema(
            fields=fields,
            description="Dynamic RAG schema"
        )

        collection_name = SCHEMA_CONFIG["collection_name"]
        print(f"📂 Creating/loading collection: {collection_name}")

        self.collection = Collection(
            name=collection_name,
            schema=schema
        )

        if index_params is None:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }

        print(f"⚙️ Creating index: {index_params}")
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print(f"✅ Collection `{collection_name}` ready!")

    def insert(self, ids, texts, filenames, page_numbers, titles, embeddings):
        """
        Insert data.
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")
        print(f"📥 Inserting {len(ids)} records...")
        self.collection.insert([ids, texts, filenames, page_numbers, titles, embeddings])
        print(f"✅ Inserted {len(ids)} records.")

    def search(self, query_embedding, top_k=3):
        """
        Search nearest vectors.
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["document"]
        )
        return results
