import torch
import torch.nn.functional as F
from typing import List

def compute_similarity(query_emb: torch.Tensor, target_emb: torch.Tensor) -> float:
    """Cosine similarity between two 1D embeddings"""
    return F.cosine_similarity(query_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()

def extract_nuggets_from_path(path_sentences: List[str],
                               query: str,
                               embed_fn,
                               top_n: int = 3):
    """
    Extract top-N relevant sentences (nuggets) from a path.
    """
    query_emb = embed_fn(query)
    sentence_embs = [embed_fn(sent) for sent in path_sentences]

    scores = [F.cosine_similarity(query_emb, emb.unsqueeze(0)).item() for emb in sentence_embs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [sentence_embs[i] for i in top_indices]

def bider_select_paths(query: str,
                       paths: List[List[str]],  # each path is a list of sentences
                       embed_fn,
                       k: int = 5,
                       lambda_: float = 0.5):
    """
    BIDER-style selection: nugget extraction + iterative gain-based refinement.
    """
    # Step 1: Extract top-3 nuggets (embeddings) per path
    query_emb = embed_fn(query)
    path_nuggets = [extract_nuggets_from_path(p, query, embed_fn, top_n=3) for p in paths]

    # Average nugget embedding as path representation
    path_embs = [torch.stack(nuggets).mean(dim=0) for nuggets in path_nuggets]

    # Step 2: Use gain-based refinement (similar to MMR)
    selected = []
    candidate_indices = list(range(len(path_embs)))

    for _ in range(k):
        best_score = -float('inf')
        best_idx = None

        for idx in candidate_indices:
            relevance = compute_similarity(query_emb, path_embs[idx])
            diversity_penalty = max(
                [compute_similarity(path_embs[idx], path_embs[j]) for j in selected]
            ) if selected else 0
            gain_score = lambda_ * relevance - (1 - lambda_) * diversity_penalty

            if gain_score > best_score:
                best_score = gain_score
                best_idx = idx

        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected
