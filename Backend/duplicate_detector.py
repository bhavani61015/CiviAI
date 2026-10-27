
from sentence_transformers import SentenceTransformer, util


# Load the model once when this module is imported.
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def find_similar_complaints(
    description: str,
    existing_complaints: list,
    category: str = "",
    threshold: float = 0.70,
    max_results: int = 5,
) -> list:
    """
    Find existing complaints with semantically similar descriptions.

    This function identifies possible duplicates; it does not
    automatically reject or delete any complaint.
    """
    if not description or not description.strip():
        return []

    new_text = f"{category.strip()}. {description.strip()}".strip(". ")
    candidates = []
    candidate_texts = []

    for complaint in existing_complaints:
        old_description = str(complaint.get("description") or "").strip()

        if not old_description:
            continue

        old_category = str(complaint.get("category") or "").strip()
        old_text = f"{old_category}. {old_description}".strip(". ")

        candidates.append(complaint)
        candidate_texts.append(old_text)

    if not candidates:
        return []

    new_embedding = model.encode(
        new_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    old_embeddings = model.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarities = util.cos_sim(new_embedding, old_embeddings)[0]

    matches = []

    for index, score in enumerate(similarities):
        similarity = float(score.item())

        if similarity >= threshold:
            complaint = candidates[index]

            matches.append({
                "complaint_id": complaint.get("complaint_id"),
                "category": complaint.get("category"),
                "description": complaint.get("description"),
                "status": complaint.get("status"),
                "similarity": round(similarity, 3),
            })

    matches.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return matches[:max_results]
