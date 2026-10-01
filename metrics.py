import torch

from utils import generate_legal_move_candidates

def compute_state_contrast_ratio(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes contrast ratio between predicted and target states.
    >1.0 indicates clear discrimination.
    ~= 1.0 indicates representation collapse.

    contrast_ratio = frac{E_{i!=j} || hat{z}_{i} - z_{j} ||_{2}}{E_{i==j} || hat{z}_{i} - z_{j} ||_{2}}

    pred_state:   (batch_size, D)
    target_state: (batch_size, D)
    """
    bs = pred_state.shape[0]
    if bs <= 1:
        return 1.0

    pred_flat = pred_state.flatten(start_dim=1)
    target_flat = target_state.flatten(start_dim=1)

    dist_matrix = torch.cdist(pred_flat, target_flat, p=2)

    # Positive distances (diagonal entries)
    pos_dists = dist_matrix.diagonal()
    mean_pos_dist = pos_dists.mean()

    # Negative distances (off-diagonal entries)
    eye_mask = torch.eye(bs, dtype=torch.bool, device=pred_state.device)
    neg_dists = dist_matrix[~eye_mask]
    mean_neg_dist = neg_dists.mean()

    contrast_ratio = mean_neg_dist / (mean_pos_dist + eps)

    return contrast_ratio

def evaluate_batch_legal_move_rankings(net, context_tokens, batch_fens, batch_target_moves, device):
    """
    Evaluates candidate legal move rankings for a batch of positions.
    """
    batch_size = len(batch_fens)

    all_candidate_tokens = []
    num_moves_per_pos = []
    target_indices = []

    # Collect all candidate move tokens across the mini-batch
    for fen, target_move in zip(batch_fens, batch_target_moves):
        cand_tokens, target_idx, legal_moves = generate_legal_move_candidates(fen, target_move)
        all_candidate_tokens.append(cand_tokens)
        num_moves_per_pos.append(len(legal_moves))
        target_indices.append(target_idx)

    # Concatenate into one single tensor: shape (total_legal_moves_in_batch, 77)
    flat_candidate_tokens = torch.cat(all_candidate_tokens, dim=0).to(device)
    context_tokens = context_tokens.to(device)

    net.eval()
    with torch.no_grad():
        # Get predicted state representations for each context
        # (Pass dummy target to extract predicted factors and state)
        context_state, pred_factors, _, _ = net(context_tokens, context_tokens)
        pred_states = net.projection.compose(pred_factors)  # Shape: (BS, state_dim)

        # Single GPU forward pass through target encoder for ALL candidate states
        # Shape: (total_legal_moves_in_batch, state_dim)
        flat_candidate_states = net.target_encoder(flat_candidate_tokens)

    # Split target states back per position and rank
    ranks = []
    percentile_ranks = []
    split_candidate_states = torch.split(flat_candidate_states, num_moves_per_pos)

    for i in range(batch_size):
        pred_state_i = pred_states[i]  # (state_dim,)
        cand_states_i = split_candidate_states[i]  # (N_i, state_dim)
        target_idx_i = target_indices[i]
        n_moves = num_moves_per_pos[i]

        if n_moves <= 1 or target_idx_i == -1:
            ranks.append(1)
            percentile_ranks.append(1.0)
            continue

        # Compute L2 distance from predicted state to each candidate target state
        distances = torch.norm(pred_state_i - cand_states_i, dim=-1) # (N_i,)

        # Find 1-indexed rank of ground-truth move
        sorted_indices = torch.argsort(distances).tolist()
        rank = sorted_indices.index(target_idx_i) + 1

        # Percentile Rank = (N - Rank) / (N - 1)
        p_rank = (n_moves - rank) / (n_moves - 1)

        ranks.append(rank)
        percentile_ranks.append(p_rank)

    return ranks, percentile_ranks
