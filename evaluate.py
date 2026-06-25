import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_app import SimpleRAG

QA_PATH = Path("data/qa.json")
RESULTS_PATH = Path("results/evaluation_results.csv")


def keyword_score(answer, keywords):
    answer_lower = answer.lower()
    matched = [word for word in keywords if word.lower() in answer_lower]
    return round(len(matched) / len(keywords), 2)


def semantic_similarity(answer, expected):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([answer, expected])
    return round(float(cosine_similarity(vectors[0], vectors[1])[0][0]), 2)


def main():
    rag = SimpleRAG()
    questions = json.loads(QA_PATH.read_text(encoding="utf-8"))
    rows = []

    for item in questions:
        answer, retrieved_docs = rag.ask(item["question"], top_k=2)
        retrieved_ids = [doc["doc_id"] for doc in retrieved_docs]

        key_score = keyword_score(answer, item["keywords"])
        sem_score = semantic_similarity(answer, item["expected_answer"])
        retrieval_hit = 1 if item["expected_doc_id"] in retrieved_ids else 0

        final_score = round((key_score * 0.4) + (sem_score * 0.4) + (retrieval_hit * 0.2), 2)

        rows.append({
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "generated_answer": answer.replace("\n", " "),
            "retrieved_doc_ids": ", ".join(retrieved_ids),
            "keyword_score": key_score,
            "semantic_similarity": sem_score,
            "retrieval_hit": retrieval_hit,
            "final_score": final_score
        })

    df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    print("Evaluation completed!")
    print(df[["question", "keyword_score", "semantic_similarity", "retrieval_hit", "final_score"]])
    print(f"\nSaved results to: {RESULTS_PATH}")
    print("\nHuman Review Rubric:")
    print("1 = Poor, 2 = Average, 3 = Good")
    print("Score each answer for: coherence, completeness, factual correctness.")


if __name__ == "__main__":
    main()
