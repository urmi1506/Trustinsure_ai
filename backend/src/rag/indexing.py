import json
import re
import html

from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_qdrant import (QdrantVectorStore,FastEmbedSparse,RetrievalMode)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


pdf_path = Path("src/data/pdfs")
markdown_path = Path("src/data/markdown")
chunk_path = Path("src/data/chunks")

def index_pdfs():
    converter= DocumentConverter()

    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Indexing {pdf_file.name}")

        result = converter.convert(pdf_file)
        markdown =result.document.export_to_markdown()
        markdown = clean_markdown(markdown)
        save_document (markdown, pdf_file)

        print(f"Extracted {len(markdown)} characters")

def clean_markdown(markdown):
    markdown = markdown.replace("<!-- image -->", "")
    markdown = html.unescape(markdown)
    markdown = re.sub(r"\s+", " ", markdown)
    markdown = re.sub(r"\s*\|\s*", " | ", markdown)
    markdown = re.sub(r"\s*`\s*", "`", markdown)
    
    return markdown.strip()

def create_chunks(markdown):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunker = SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

    return chunker.split_text(markdown)

# def create_embedding(chunks):
#     # dense embedding
#     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     dense_embeddings = embedding.embed_documents(chunks)

#     # sparse embedding
#     sparse_tokens = [chunk.lower().split() for chunk in chunks]

#     return dense_embeddings, sparse_tokens


def create_faiss_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store


def create_qdrant_store(chunks,source_name):

    # Dense embedding
    dense_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Sparse embedding
    sparse_embedding = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

    # Create collection + vector store
    vector_store = QdrantVectorStore.construct_instance(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name="documents",

        client_options={
            "url": "http://localhost:6333"
        }
    )

    # Add chunks
    vector_store.add_texts(
        texts=chunks,
        metadatas=[
            {"source": source_name}
            for _ in chunks
        ]
    )
    return vector_store


def create_vector_store(chunks, source_name, store_type="qdrant"):
    if store_type == "faiss":
        return create_faiss_store(chunks)
    elif store_type == "qdrant":
        return create_qdrant_store(chunks, source_name)
    else:
        raise ValueError(
            f"Unsupported vector store: {store_type}"
        )


def save_document(markdown, pdf_file):
    markdown_output = markdown_path/f"{pdf_file.stem}.md"
    markdown_output.write_text(markdown, encoding="utf-8")

    print(f"Saved: {markdown_output}")

    # create chunks
    chunks = create_chunks(markdown)

    # save chunks
    chunk_file = chunk_path/f"{pdf_file.stem}.json"

    data = [ { "chunk_id": i, "text": chunk,
        } 
        for i, chunk in enumerate(chunks) ]

    chunk_file.write_text(json.dumps(data ,indent=2 ,ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {chunk_file}")
    print(f"Created {len(chunks)} chunks")

    # store in Qdrant 

    vector_store = create_vector_store(
        chunks,
        pdf_file.stem,
        store_type="qdrant"
    )
    print(f"Stored {len(chunks)} chunks in Qdrant vector store")
    
if __name__ == "__main__":
    index_pdfs()