"""
Core module for evaluation methods like the masking game.
"""
import torch

# ===================================================================
# Batch-processed versions of the functions
# ===================================================================

def topk_jaccard_batch(clean_rank, noisy_rank, k):
    """
    Computes Top-K Jaccard similarity for a batch of samples.
    Input clean_rank shape: [B, H, W]
    Input noisy_rank shape: [B, H, W]
    Input k: int
    Output: a 1D Tensor of shape [B] containing float scores
    """
    num_total_patches = clean_rank.shape[1] * clean_rank.shape[2]
    if k == 0:
        return torch.ones(clean_rank.shape[0], device=clean_rank.device)
    if k > num_total_patches:
        k = num_total_patches

    clean_flat = clean_rank.flatten(start_dim=1)
    noisy_flat = noisy_rank.flatten(start_dim=1)

    clean_topk_indices = torch.topk(clean_flat, k, dim=1, largest=False).indices
    noisy_topk_indices = torch.topk(noisy_flat, k, dim=1, largest=False).indices

    comparison_matrix = clean_topk_indices.unsqueeze(2) == noisy_topk_indices.unsqueeze(1)
    intersection_size = comparison_matrix.any(dim=2).sum(dim=1).float()
    union_size = (2.0 * k) - intersection_size
    
    jaccard_scores = torch.where(
        union_size > 0,
        intersection_size / union_size,
        torch.ones_like(intersection_size)
    )
    return jaccard_scores

# ===================================================================
# Masking Game Functions (operate on batches)
# ===================================================================

def apply_mask_batch(images, rank_matrix, k, patch_size, replace_top, fill_tensors_batch):
    """
    Applies a mask to a batch of images based on a rank matrix.
    k can be an integer or a tensor of shape [B].
    """
    B, _, H, W = images.shape
    num_patches_total = rank_matrix.shape[-2] * rank_matrix.shape[-1]

    if isinstance(k, int) and k == 0:
        return images.clone()
    
    flat_ranks = rank_matrix.flatten(start_dim=1)
    
    # Get sorted indices once. `descending=False` means lower rank is more important.
    sorted_indices = torch.argsort(flat_ranks, dim=1, descending=not replace_top)
    
    # Create a mask for each sample in the batch based on its k value
    aranged_tensor = torch.arange(num_patches_total, device=images.device).expand(B, -1)
    
    k_tensor = k if torch.is_tensor(k) else torch.tensor([k] * B, device=images.device)
    k_mask = aranged_tensor < k_tensor.unsqueeze(1)
    
    patches_to_mask_flat = torch.zeros_like(flat_ranks, dtype=torch.bool)
    patches_to_mask_flat.scatter_(1, sorted_indices, k_mask)
    
    patches_to_mask = patches_to_mask_flat.view_as(rank_matrix)
    
    # Upscale the patch mask to the full image resolution
    full_res_mask = patches_to_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    
    if full_res_mask.dim() == 3:
        full_res_mask = full_res_mask.unsqueeze(1)
    if full_res_mask.size(1) != images.size(1):
        full_res_mask = full_res_mask.expand(-1, images.size(1), -1, -1)
    
    return torch.where(full_res_mask, fill_tensors_batch, images)

def evaluate_batch(masked_preds, original_preds, labels):
    """
    Evaluates metrics for a batch of masked predictions.
    Returns Tensors instead of scalars to support asynchronous GPU operations.
    """
    accuracy = (masked_preds == labels).sum()
    consistency = (masked_preds == original_preds).sum()
    return accuracy, consistency
