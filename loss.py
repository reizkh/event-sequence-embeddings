from typing import List
import torch


def contrastive_loss_euclidean(
    ids: List[int],
    embeddings: torch.Tensor,
    margin: float = 0.5
):
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"Несоответствие размеров: len(ids)={len(ids)}, "
            f"latents.shape[0]={embeddings.shape[0]}"
        )
    ids_tensor = torch.tensor(ids, device=embeddings.device)
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    positive_mask = (ids_tensor.unsqueeze(0) == ids_tensor.unsqueeze(1)) * 1.0
    negative_mask = 1 - positive_mask

    positive_term = torch.sum(positive_mask * distances**2)
    negative_term = torch.sum(negative_mask * torch.clamp(margin - distances, min=0.0)**2)

    return 0.5 * (positive_term + negative_term)