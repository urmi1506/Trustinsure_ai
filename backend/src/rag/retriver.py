import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.rag.vector_store import get_vector_store

# Load environment variables
load_dotenv()

# Qdrant
vector_store = get_vector_store()

# Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
# Result folder
result_path = Path("src/data/results")

def retrieve(query, k=5):
    return vector_store.similarity_search_with_score(
        query,
        k=k
    )

def generate_answer(query,history=None, k=5):
    documents = retrieve(query, k)
    context = "\n\n".join(
        document.page_content
        for document, score in documents
    )
    if history is None:
        history = []

    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )

    prompt = f"""
You are a health insurance assistant.

Answer the user's question using only the provided context.

IMPORTANT RULES:
1. Never invent or calculate an exact premium unless the context explicitly provides
   the exact premium for the user's specific situation.
2. If the exact answer is not available, do NOT simply say "I don't have enough
   information" if the context contains a relevant example or related information.
3. Clearly distinguish between an example and the user's specific situation.
4. If the document provides a premium example with different conditions, mention
   that example and explain why it cannot be used as the exact answer.
5. If the document explicitly says that the exact premium must be calculated
   separately, mention that.
6. Use only the provided context and conversation history.
7. Do not use outside knowledge.

Conversation history:
{conversation}

Context:
{context}

Current question:
{query}

Answer:
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return documents, response.text

def save_result(query, documents, answer):

    result = {
        "question": query,
        "retrieved_chunks": [],
        "generated_answer": answer
    }

    for rank, (document, score) in enumerate(
        documents,
        start=1
    ):
        result["retrieved_chunks"].append({
            "rank": rank,
            "retrieval_score": round(score, 4),
            "source": document.metadata.get("source"),
            "chunk": document.page_content
        })

    output_file = result_path / "rag_results.json"

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = []

    results.append(result)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    print("Enter questions. Type 'exit' to stop.\n")

    while True:

        query = input("Question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        documents, answer = generate_answer(query)

        print("\nAnswer:")
        print(answer)

        # print("\nRetrieved chunks:\n")

        # for rank, (document, score) in         enumerate(documents, start=1):
        #     print(f"\n--- Chunk {rank} |         Score: {score} ---")
        #     print(document.page_content)

        save_result(
                    query,
                    documents,
                    answer
                )