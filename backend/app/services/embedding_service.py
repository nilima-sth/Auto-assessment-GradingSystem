from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import get_settings
from backend.app.services.document_service import chunk_text


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the sentence embedding model once."""
    return SentenceTransformer(get_settings().embedding_model)


def map_answers_to_questions(
    questions: dict[str, dict],
    ocr_text: str,
    chunk_size: int = 200,
    overlap: int = 50,
    threshold: float = 0.6,
) -> dict[str, str]:
    """Map OCR chunks to the most semantically similar exam question."""
    chunks = chunk_text(ocr_text, chunk_size=chunk_size, overlap=overlap)
    question_keys = list(questions.keys())
    question_texts = [question["text"] for question in questions.values()]
    question_embeddings = get_embedder().encode(question_texts)
    mapped = {key: "" for key in question_keys}
    flagged = []

    for chunk in chunks:
        chunk_embedding = get_embedder().encode([chunk])
        similarities = cosine_similarity(chunk_embedding, question_embeddings)[0]
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        if best_similarity >= threshold:
            mapped[question_keys[best_index]] += chunk + " "
        else:
            flagged.append(chunk)

    if flagged:
        mapped["Flagged"] = " ".join(flagged)
    return mapped
