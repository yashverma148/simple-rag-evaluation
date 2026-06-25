# Simple Custom RAG & Evaluation Assignment

## Domain
This project uses a small specialized dataset on **Wheat Disease and Farm Advisory**.
The dataset contains 5 short documents about yellow rust, loose smut, irrigation, nitrogen management, and weed control.

## Project Structure

```text
simple_rag_assignment/
├── rag_app.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   ├── docs.json
│   └── qa.json
└── results/
```

## RAG Pipeline Design

1. **Load documents** from `data/docs.json`.
2. **Create embeddings** using TF-IDF vectorization.
3. **Retrieve relevant documents** using cosine similarity.
4. **Generate answer** using the best retrieved document as context.

This is a simple and lightweight RAG system. It does not require paid API keys.

## How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Ask a question

```bash
python rag_app.py --ask "What are the symptoms of yellow rust in wheat?"
```

### 3. Run evaluation

```bash
python evaluate.py
```

The evaluation result will be saved in:

```text
results/evaluation_results.csv
```

## Evaluation Framework

The evaluation uses 3 simple metrics:

### 1. Keyword Score
Checks how many important expected keywords are present in the generated answer.

Formula:

```text
matched keywords / total keywords
```

### 2. Semantic Similarity
Uses TF-IDF cosine similarity between generated answer and expected answer.

### 3. Retrieval Hit
Checks whether the correct source document was retrieved in top-2 results.

```text
1 = correct document retrieved
0 = correct document not retrieved
```

### Final Score

```text
Final Score = 0.4 × Keyword Score + 0.4 × Semantic Similarity + 0.2 × Retrieval Hit
```

## Human Qualitative Assessment

A human reviewer can score each answer from 1 to 3:

| Criterion | Meaning |
|---|---|
| Coherence | Answer is clear and readable |
| Completeness | Answer covers the main points |
| Factual Correctness | Answer matches the source document |

Scoring:

```text
1 = Poor
2 = Average
3 = Good
```

## Challenges and Lessons Learned

The main challenge was creating a small but meaningful dataset and designing evaluation metrics that are simple but useful. This project shows that even a lightweight RAG system can be evaluated using retrieval accuracy, keyword matching, and semantic similarity.

## Video Demo Plan

1. Explain the domain and dataset.
2. Show `rag_app.py` and ask 2 questions.
3. Show retrieved documents and generated answer.
4. Run `python evaluate.py`.
5. Explain keyword score, semantic similarity, retrieval hit, and final score.
