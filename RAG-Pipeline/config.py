# config.py

SCHEMA_CONFIG = {
    "collection_name": "RAG_Pipeline_Test_Git",
    "fields": [
        {
            "name": "id",
            "dtype": "VARCHAR",
            "is_primary": True,
            "max_length": 100
        },
        {
            "name": "document",
            "dtype": "VARCHAR",
            "max_length": 8192
        },
        {
            "name": "filename",
            "dtype": "VARCHAR",
            "max_length": 512
        },
        {
            "name": "page_numbers",
            "dtype": "VARCHAR",
            "max_length": 128
        },
        {
            "name": "title",
            "dtype": "VARCHAR",
            "max_length": 512
        },
        {
            "name": "embedding",
            "dtype": "FLOAT_VECTOR",
            "dim": "512"  # replace this with self.EMBED_DIM
        }
    ]
}
