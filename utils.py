import chess
import torch

import torch.optim as optim
from torch.utils.data import DataLoader

import tokenizer
from config import TrainConfig
from model import ChessOPFModel
from encoder import ChessEncoder
from dataset import ChessDataset

def get_net(config: TrainConfig, device):
    enc = ChessEncoder(
        vocab_size=config.enc_vocab_size,
        max_sequence_length=config.enc_max_sequence_length,
        embedding_dim=config.enc_embedding_dim,
        num_layers=config.enc_num_layers,
        num_heads=config.enc_num_heads,
        hidden_act=config.enc_hidden_act,
        hidden_dropout_prob=config.enc_hidden_dropout_prob,
        attention_probs_dropout_prob=config.enc_attention_probs_dropout_prob,
    ).to(device)

    net = ChessOPFModel(
        context_encoder=enc,
        target_encoder=None,
        state_dim=config.state_dim,
        num_factors=config.num_factors,
        factor_dim=config.factor_dim,
        hidden_dim=config.hidden_dim,
        target_momentum=config.target_momentum,
    ).to(device)

    return net

# TODO: Support more optimizers
def get_optimizer(net: ChessOPFModel, config: TrainConfig):
    optimizer = optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return optimizer

def get_dataloaders(config: TrainConfig):
    train_loader = DataLoader(
        ChessDataset(config.train_data_dir),
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
    )
    test_loader = DataLoader(
        ChessDataset(config.test_data_dir),
        batch_size=config.batch_size,
        num_workers=4,
    )

    return train_loader, test_loader


def generate_legal_move_candidates(fen: str, target_move_uci: str | None = None):
    """
    Generates tokenized target states for all legal moves from a given FEN.

    Args:
        fen: The current position FEN string.
        target_move_uci: Optional UCI string of the ground-truth move (e.g. "e2e4").

    Returns:
        candidate_tokens: Tensor of shape (num_legal_moves, 77)
        target_idx: Index of the ground-truth move in candidate_tokens (-1 if not provided/found)
        legal_moves: List of chess.Move objects corresponding to rows in candidate_tokens
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    candidate_tokens_list = []
    target_idx = -1

    for i, move in enumerate(legal_moves):
        uci_str = move.uci()
        if target_move_uci is not None and uci_str == target_move_uci:
            target_idx = i

        # Push move in-place
        board.push(move)
        next_fen = board.fen()
        candidate_tokens_list.append(tokenizer.tokenize(next_fen))
        # Pop move back to restore board state
        board.pop()

    candidate_tokens = torch.stack(candidate_tokens_list, dim=0)  # Shape: (N_moves, 77)

    return candidate_tokens, target_idx, legal_moves
