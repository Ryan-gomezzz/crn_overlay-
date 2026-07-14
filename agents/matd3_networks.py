import torch
import torch.nn as nn
import torch.nn.functional as F


class MAActorNetwork(nn.Module):
    def __init__(self, obs_dim: int = 8, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        # We need an encoder similar to GRUBeliefEncoder for each agent's local history
        # Input to encoder is 11D (obs_dim + action_dim(1) + decoded(1) + outage(1))
        self.input_dim = obs_dim + action_dim + 2
        
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU()
        )
        self.gru = nn.GRU(32, 64, num_layers=1, batch_first=True)
        
        self.actor = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )

    def forward(self, hist_seq: torch.Tensor) -> torch.Tensor:
        """
        hist_seq: (B, L, 11)
        """
        B, L, D = hist_seq.shape
        x = hist_seq.reshape(B * L, D)
        embedded = self.embed(x)
        embedded = embedded.reshape(B, L, 32)
        
        gru_out, _ = self.gru(embedded)
        belief = gru_out[:, -1, :]  # (B, 64)
        
        return self.actor(belief)
        
    def get_belief(self, hist_seq: torch.Tensor) -> torch.Tensor:
        B, L, D = hist_seq.shape
        x = hist_seq.reshape(B * L, D)
        embedded = self.embed(x)
        embedded = embedded.reshape(B, L, 32)
        gru_out, _ = self.gru(embedded)
        return gru_out[:, -1, :]


class MACriticNetwork(nn.Module):
    def __init__(self, num_agents: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Centralized critic takes beliefs of all agents + actions of all agents + relay action
        # State dim = 64 * num_agents
        
        self.state_dim = 64 * num_agents
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state: (B, 64*N)
        action: (B, N+1)
        """
        xu = torch.cat([state, action], dim=1)
        return self.q1(xu)
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=1)
        return self.q1(xu), self.q2(xu)


class CentralizedRelayActor(nn.Module):
    def __init__(self, num_agents: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        # Takes all agents' beliefs (N * 64) as input
        self.input_dim = num_agents * 64
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )

    def forward(self, global_belief: torch.Tensor) -> torch.Tensor:
        """
        global_belief: (B, N * 64)
        """
        return self.actor(global_belief)

