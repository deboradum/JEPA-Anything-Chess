import torch.optim as optim
from torch.utils.data import DataLoader

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
