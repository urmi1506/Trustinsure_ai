import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.dataset_schema import SingleTurnSample


load_dotenv()

with open("src/data/test-que/test_questions.json", encoding="utf-8") as f:
    tests = json.load(f)

with open("src/data/results/rag_results.json", encoding="utf-8") as f:
    results = json.load(f)


def get_llm():

    provider = os.getenv("EVAL_LLM_PROVIDER", "gemini")

    if provider == "gemini":

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        return llm_factory(
            "gemini-2.5-flash",
            provider="google",
            client=client
        )

    elif provider == "groq":

        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        return llm_factory(
            "openai/gpt-oss-120b",
            provider="openai",
            client=client
        )

    else:
        raise ValueError(
            f"Unsupported EVAL_LLM_PROVIDER: {provider}"
        )


llm = get_llm()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(
    llm=llm,
    embeddings=embeddings
)
context_precision = ContextPrecision(llm=llm)
context_recall = ContextRecall(llm=llm)


async def evaluate():

    evaluation_results = []

    for i, (test, result) in enumerate(zip(tests, results), 1):

        contexts = [
            chunk["chunk"]
            for chunk in result["retrieved_chunks"]
        ]

        question = test["question"]
        answer = result["generated_answer"]
        reference = test["ground_truth"]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=reference,
            retrieved_contexts=contexts
        )

        faith = await faithfulness.single_turn_ascore(sample)
        relevancy = await answer_relevancy.single_turn_ascore(sample)
        precision = await context_precision.single_turn_ascore(sample)
        recall = await context_recall.single_turn_ascore(sample)

        evaluation_results.append({
            "question": question,
            "faithfulness": round(faith, 2),
            "answer_relevancy": round(relevancy, 2),
            "context_precision": round(precision, 2),
            "context_recall": round(recall, 2)
        })

        print(f"\nQ{i}")
        print(f"Faithfulness:      {faith:.2f}")
        print(f"Answer Relevancy:  {relevancy:.2f}")
        print(f"Context Precision: {precision:.2f}")
        print(f"Context Recall:    {recall:.2f}")

    averages = {
        "faithfulness": round(
            sum(x["faithfulness"] for x in evaluation_results)
            / len(evaluation_results), 2
        ),
        "answer_relevancy": round(
            sum(x["answer_relevancy"] for x in evaluation_results)
            / len(evaluation_results), 2
        ),
        "context_precision": round(
            sum(x["context_precision"] for x in evaluation_results)
            / len(evaluation_results), 2
        ),
        "context_recall": round(
            sum(x["context_recall"] for x in evaluation_results)
            / len(evaluation_results), 2
        )
    }

    output = {
        "results": evaluation_results,
        "average_scores": averages
    }

    with open(
        "src/data/results/evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=2)

    print("\nEvaluation saved to evaluation_results.json")
    print("\nAverage Scores:")
    for name, score in averages.items():
        print(f"{name}: {score:.2f}")


asyncio.run(evaluate())