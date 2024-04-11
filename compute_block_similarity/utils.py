import torch
from typing import List

def angular_distance(x_l, x_l_plus_n) -> torch.Tensor:
    """Compute the angular distance between layer output tokens."""
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi

def compute_block_distances(hidden_states: List[torch.Tensor], layers_to_skip: int) -> List[float]:
    """Compute and return angular distances for each block of layers."""
    distances = []
    num_layers = len(hidden_states)
    for l in range(num_layers - layers_to_skip):
        block_distance = angular_distance(hidden_states[l], hidden_states[l + layers_to_skip]).mean().item()
        distances.append(block_distance)
    return distances

def get_last_non_padded_tokens(hidden_states, attention_mask) -> List[torch.Tensor]:
    """Get last non-padded tokens for each layer."""
    last_non_padded_hidden_states = []
    for layer in hidden_states:
        batch_size, _, _ = layer.size()
        batch_last_tokens = []
        for batch in range(batch_size):
            last_non_pad_index = attention_mask[batch].nonzero(as_tuple=True)[0].max()
            last_token = layer[batch, last_non_pad_index, :]
            batch_last_tokens.append(last_token.unsqueeze(0))
        last_non_padded_hidden_states.append(torch.cat(batch_last_tokens, dim=0))
    return last_non_padded_hidden_states