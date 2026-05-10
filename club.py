import torch
from torch import nn


class CLUB(nn.Module):
    def __init__(
            self,
            emb_dim: int = 128,
            hidden_dim: int = 128
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n = z1.shape[0]

        pairs = torch.concat([z1.unsqueeze(0).expand(n, -1, -1), z2.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        log_q = self.net(pairs)
        log_q = log_q.squeeze(-1)
        return log_q
