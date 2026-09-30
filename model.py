import torch
import torch.nn as nn
from torch import Tensor

from jepa_anything_core.baselines import _BaseJEPABaseline
from jepa_anything_core.opf import validate_factorization, OrthogonalFactorProjection

class ChessOPFModel(_BaseJEPABaseline):
    """JEPA-Anything chess model."""

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module | None = None,
        state_dim: int = 256,
        num_factors: int = 16,
        factor_dim: int = 16,
        hidden_dim: int = 512,
        target_momentum: float = 0.996,
    ):
        # Ensure d = K * r
        validate_factorization(state_dim, num_factors, factor_dim)

        super().__init__(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            target_momentum=target_momentum,
        )

        self.state_dim = state_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        # Predictor mapping context state s_t -> factor predictions (K, r)
        self.predictor_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # K independent heads for each factor subspace
        self.factor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, factor_dim) for _ in range(num_factors)
        ])

        # Learned projection maps for orthogonal factor analysis (d x r per factor)
        self.projection = OrthogonalFactorProjection(
            state_dim=state_dim,
            num_factors=num_factors,
            factor_dim=factor_dim,
            learnable=True,
            orthogonality_mode="soft_gram"
        )

    def forward(
        self,
        context_tokens: Tensor,
        target_tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass outputting context latent, predicted factors, target latent, and projected target factors."""

        # Encode context state s_t and target state y_{t+1} (target handles stop-gradient)
        context_state, target_state = self._encode(context_tokens, target_tokens)

        # Predict target factors z_hat_{t+1} with trailing shape (Batch, K, r)
        hidden = self.predictor_trunk(context_state)
        pred_factors = torch.stack([head(hidden) for head in self.factor_heads], dim=1)

        # Project target latent y_{t+1} onto orthogonal factor subspaces
        target_factors = self.projection.decompose(target_state)

        return context_state, pred_factors, target_state, target_factors
