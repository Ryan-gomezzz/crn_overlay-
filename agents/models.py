"""
Neural Network Architectures for TD3 and CAMO-TD3.
"""

import torch
import torch.nn as nn


class GRUBeliefEncoder(nn.Module):
    """
    GRU Belief Encoder to process sequential observations and actions.
    Maps (s_t-L:t, a_t-L:t-1, ...) -> b_t
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        input_dim: int = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input is either obs + action or custom input_dim
        if input_dim is None:
            input_dim = obs_dim + action_dim

        # Embedding layer
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )

        # GRU Layer
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            obs_seq: Tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, 
            seq_len, input_dim)
            act_seq: Optional tensor of shape (batch_size, seq_len, action_dim)
        Returns:
            Belief state b_t of shape (batch_size, hidden_dim)
        """
        # Concat along features dimension if act_seq is provided
        if act_seq is not None:
            inputs = torch.cat([obs_seq, act_seq], dim=-1)
        else:
            inputs = obs_seq

        embeddings = self.embed(inputs)  # (batch, seq_len, embed_dim)

        # Pass through GRU. We use zero initial hidden state h_0
        gru_out, _ = self.gru(embeddings)  # gru_out shape: (batch, seq_len, hidden_dim)

        # Expose the final step's hidden state as the current belief state b_t
        belief = gru_out[:, -1, :]  # Shape: (batch, hidden_dim)
        return belief


class CAMO_Actor(nn.Module):
    """
    CAMO Actor Network. Maps belief state b_t -> action a_t.
    """

    def __init__(self, belief_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # Actions are power fractions in [0, 1]
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.net(belief)


class TwinCritics(nn.Module):
    """
    Twin Critic Networks. Evaluates Q_1(s, a) and Q_2(s, a) to avoid overestimation 
    bias.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns Q1(state, action)
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Q1, Q2)
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# --- STANDARD TD3 NETWORKS (BACKWARD COMPATIBILITY) ---


class TD3_Actor(nn.Module):
    """
    Standard TD3 Actor Network. Maps state observation s_t -> action a_t.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
