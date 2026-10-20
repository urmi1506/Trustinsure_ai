from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from qdrant_client import QdrantClient


COLLECTION_NAME = "documents"
QDRANT_URL = "http://localhost:6333"


dense_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

sparse_embedding = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

client = QdrantClient(
    url=QDRANT_URL
)


def get_vector_store():

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID
    )