import bagz
import chess
import torch

import tokenizer
import constants

from torch.utils.data import Dataset


def _process_fen(fen: str) -> torch.Tensor:
  return tokenizer.tokenize(fen)

class ChessDataset(Dataset):
    def __init__(self, path):
        self.data_source = bagz.BagDataSource(path=path)
        self.num_records = len(self.data_source)

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        element = self.data_source[idx]

        fen, move = constants.CODERS['behavioral_cloning'].decode(element)

        state = _process_fen(fen)
        board = chess.Board(fen)

        board.push(chess.Move.from_uci(move))
        next_fen = board.fen()
        next_state = _process_fen(next_fen)

        return state, next_state  # (context, target)
