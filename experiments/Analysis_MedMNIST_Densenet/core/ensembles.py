"""
Core module for all ensemble methods.
"""
import torch

# ===================================================================
# Batch-processed versions of the functions
# ===================================================================

# --- Rank-based batch functions ---

def borda_batch(rank_stack):
    """Borda count for a batch of samples. Input shape: [B, E, H, W]"""
    total_scores = rank_stack.float().sum(dim=1)
    flat_scores = total_scores.flatten(start_dim=1)
    final_ranks = torch.argsort(torch.argsort(flat_scores, dim=1, descending=False), dim=1, descending=False)
    return final_ranks.reshape(total_scores.shape)

def rrf_batch(rank_stack, k):
    """Reciprocal Rank Fusion for a batch of samples. Input shape: [B, E, H, W]"""
    rrf_scores = 1.0 / (float(k) + rank_stack.float())
    total_scores = rrf_scores.sum(dim=1)
    flat_scores = total_scores.flatten(start_dim=1)
    final_ranks = torch.argsort(torch.argsort(flat_scores, dim=1, descending=True), dim=1, descending=False)
    return final_ranks.reshape(total_scores.shape)

def schulze_batch(rank_stack):
    """Schulze method for a batch of samples. Input shape: [B, E, H, W]"""
    B, E, H_p, W_p = rank_stack.shape
    num_patches = H_p * W_p
    ranks_flat = rank_stack.flatten(start_dim=2)
    
    pairwise_pref = (ranks_flat.unsqueeze(3) < ranks_flat.unsqueeze(2)).to(rank_stack.dtype).sum(dim=1)
    
    strongest_paths = torch.where(pairwise_pref > pairwise_pref.transpose(1, 2), pairwise_pref, torch.zeros_like(pairwise_pref))
    
    for k_iter in range(num_patches):
        path_through_k = torch.min(strongest_paths[:, :, k_iter].unsqueeze(2), strongest_paths[:, k_iter, :].unsqueeze(1))
        strongest_paths = torch.max(strongest_paths, path_through_k)
        
    wins_count = (strongest_paths > strongest_paths.transpose(1, 2)).to(rank_stack.dtype).sum(dim=2)
    final_ranks = torch.argsort(torch.argsort(wins_count, dim=1, descending=True), dim=1, descending=False)
    return final_ranks.view(B, H_p, W_p)

def kemeny_young_batch(rank_stack):
    """Kemeny-Young method for a batch of samples. Input shape: [B, E, H, W]"""
    B, E, H_p, W_p = rank_stack.shape
    ranks_flat = rank_stack.flatten(start_dim=2)
    
    pairwise_pref = (ranks_flat.unsqueeze(3) < ranks_flat.unsqueeze(2)).to(rank_stack.dtype).sum(dim=1)
    score_matrix = pairwise_pref - pairwise_pref.transpose(1, 2)
    
    total_scores = score_matrix.sum(dim=2)
    final_ranks = torch.argsort(torch.argsort(total_scores, dim=1, descending=True), dim=1, descending=False)
    return final_ranks.view(B, H_p, W_p)

# --- Norm-based batch functions ---

def _normal_standardization_batch(attribution_map):
    """Normal standardization (z-score). Input shape: [B, C, H, W]"""
    mean = torch.mean(attribution_map, dim=(-1, -2), keepdim=True)
    std = torch.std(attribution_map, dim=(-1, -2), keepdim=True)
    return (attribution_map - mean) / (std + 1e-8)

def _robust_standardization_batch(attribution_map):
    """Robust standardization using median and IQR. Input shape: [B, C, H, W]"""
    B, C, H, W = attribution_map.shape
    attribution_map_float = attribution_map.float().flatten(start_dim=1)
    
    q1 = torch.quantile(attribution_map_float, 0.25, dim=1, keepdim=True)
    q3 = torch.quantile(attribution_map_float, 0.75, dim=1, keepdim=True)
    median = torch.median(attribution_map_float, dim=1, keepdim=True).values
    
    iqr = q3 - q1
    
    # Reshape for broadcasting
    median = median.view(B, 1, 1, 1)
    iqr = iqr.view(B, 1, 1, 1)
    
    return (attribution_map - median) / (iqr + 1e-8)

def _scaling_by_second_moment_batch(attribution_map):
    """Scaling by the average second moment estimate. Input shape: [B, C, H, W]"""
    channel_stds = torch.std(attribution_map, dim=(-1, -2))
    avg_second_moment = torch.mean(channel_stds, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return attribution_map / (avg_second_moment + 1e-8)

# ===================================================================
# Main Ensemble Logic
# ===================================================================

_NORMALIZATION_FUNCTIONS = {
    "normal": _normal_standardization_batch,
    "robust": _robust_standardization_batch,
    "scaling": _scaling_by_second_moment_batch,
}

def norm_ensemble_batch(attribution_stack, normalization, aggregation):
    """
    Runs the specified norm+aggregation ensemble method on a batch of attribution stacks.

    Args:
        attribution_stack (torch.Tensor): Shape [B, N, C, H, W], where B is batch size,
                                          N is the number of methods to ensemble.
        normalization (str): Type of normalization ('normal', 'robust', 'scaling').
        aggregation (str): Type of aggregation ('mean', 'max', 'min').

    Returns:
        torch.Tensor: The ensembled attribution map of shape [B, C, H, W].
    """
    if normalization not in _NORMALIZATION_FUNCTIONS:
        raise ValueError(f"Unknown normalization type: {normalization}")
    
    B, N, C, H, W = attribution_stack.shape
    reshaped_stack = attribution_stack.view(-1, C, H, W)

    norm_func = _NORMALIZATION_FUNCTIONS[normalization]
    normalized_attributions = norm_func(reshaped_stack)

    normalized_stack = normalized_attributions.view(B, N, C, H, W)

    if aggregation == "mean":
        return torch.mean(normalized_stack, dim=1)
    elif aggregation == "max":
        return torch.max(normalized_stack, dim=1).values
    elif aggregation == "min":
        return torch.min(normalized_stack, dim=1).values
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation}")

# ===================================================================
# Factory to get ensemble functions
# ===================================================================

class EnsembleFactory:
    """
    A factory class to retrieve the correct ensemble function based on config.
    """
    @staticmethod
    def get_ensemble_method(method_config: dict):
        """
        Returns a callable function for the specified ensemble method.

        Args:
            method_config (dict): The configuration for the method, e.g.,
                                  {'name': 'RRF', 'params': {'k': 60}} or
                                  {'name': 'norm_ensemble', 'params': {'normalization': 'robust', 'aggregation': 'mean'}}

        Returns:
            A tuple of (callable, bool): The function and a boolean indicating if it operates on ranks (True) or values (False).
        """
        name = method_config['name']
        params = method_config.get('params', {})

        if name == 'Borda':
            return borda_batch, True
        elif name == 'RRF':
            return rrf_batch, True
        elif name == 'Schulze':
            return schulze_batch, True
        elif name == 'Kemeny-Young':
            return kemeny_young_batch, True
        elif name == 'norm_ensemble':
            return lambda values: norm_ensemble_batch(values, **params), False
        else:
            raise ValueError(f"Unknown ensemble method: {name}")