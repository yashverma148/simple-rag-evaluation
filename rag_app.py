import json
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/docs.json")


class SimpleRAG:
    def __init__(self, docs_path=DATA_PATH):
        self.docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        self.texts = [doc["title"] + ". " + doc["text"] for doc in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=2):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        ranked_indexes = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indexes:
            doc = self.docs[idx]
            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(float(scores[idx]), 4)
            })
        return results

    def generate_answer(self, query, retrieved_docs):
        # Simple extractive generation: answer from the best retrieved document.
        best_doc = retrieved_docs[0]
        return (
            f"Answer: {best_doc['text']}\n\n"
            f"Source: {best_doc['title']} ({best_doc['doc_id']})"
        )

    def ask(self, query, top_k=2):
        retrieved_docs = self.retrieve(query, top_k=top_k)
        answer = self.generate_answer(query, retrieved_docs)
        return answer, retrieved_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG App")
    parser.add_argument("--ask", required=True, help="Ask a question")
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()

    rag = SimpleRAG()
    answer, docs = rag.ask(args.ask, top_k=args.top_k)

    print("\nQuestion:", args.ask)
    print("\n" + answer)
    print("\nRetrieved Chunks:")
    for doc in docs:
        print(f"- {doc['doc_id']} | {doc['title']} | score={doc['score']}")
