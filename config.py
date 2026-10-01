import yaml  # type: ignore
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 123
    run_name: str = "You forgot to change the run name"

    # Encoder args
    enc_vocab_size: int = 32
    enc_max_sequence_length: int = 128
    enc_embedding_dim: int = 256
    enc_num_layers: int = 4
    enc_num_heads: int = 4
    enc_hidden_act: str = "gelu"
    enc_hidden_dropout_prob: float = 0.1
    enc_attention_probs_dropout_prob: float = 0.1

    # Model args
    state_dim: int = 256
    num_factors: int = 16
    factor_dim: int = 16
    hidden_dim: int = 512
    target_momentum: float = 0.996

    # Loss args
    orthogonality_weight: float = 1.0
    factor_activity_weight: float = 1.0
    encoder_variance_weight: float = 1.0

    # Train args
    train_data_dir: str = "data/train"
    test_data_dir: str = "data/test"
    epochs: int = 2
    optimizer: str = "adamW"
    beta_2: float = 0.95
    log_interval: int = 100
    eval_interval: int = 1000
    learning_rate: float = 0.0001
    weight_decay: float = 0.05
    batch_size: int = 64
    gradient_clipping_norm: float = 1.0
    early_stop: int = 3

    pretrained_path: str = ""

def load_config(yaml_path: str) -> TrainConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)
