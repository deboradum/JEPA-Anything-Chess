import copy
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel

class ChessEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=32,  # vocab_size=utils.NUM_ACTIONS,
        max_sequence_length=128,  # tokenizer.SEQUENCE_LENGTH + 2,
        embedding_dim=256,
        num_layers=4,
        num_heads=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        super().__init__()

        self.encoder = BertModel(BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_sequence_length,
            hidden_size=embedding_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4*embedding_dim,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        ))


    def forward(self, input_tokens, attention_mask=None):
        input_tokens = input_tokens.to(dtype=torch.int32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.int32)

        outputs = self.encoder(
            input_ids=input_tokens,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        return hidden_states.mean(dim=1)
