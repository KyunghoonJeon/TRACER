import torch
import torch.nn.functional as F
from typing import List

def compute_similarity(query_emb: torch.Tensor, target_emb: torch.Tensor) -> float:
    """Cosine similarity between two 1D embeddings"""
    return F.cosine_similarity(query_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()

def mmr_select_paths(query_emb: torch.Tensor,
                     path_embs: List[torch.Tensor],
                     k: int = 5,
                     lambda_: float = 0.5) -> List[int]:
    """
    Maximal Marginal Relevance based selection of k paths.
    """
    selected = []
    candidate_indices = list(range(len(path_embs)))

    for _ in range(k):
        best_score = -float('inf')
        best_idx = None

        for idx in candidate_indices:
            relevance = compute_similarity(query_emb, path_embs[idx])
            redundancy = max(
                [compute_similarity(path_embs[idx], path_embs[j]) for j in selected]
            ) if selected else 0
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected
