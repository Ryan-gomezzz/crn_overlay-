import torch
import torch.nn as nn
import torch.nn.functional as F


class CentActorNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        # Takes all agents' histories at once
        # Input features per agent = obs_dim + 1(action) + 1(decode) + 1(outage) = 11
        self.num_agents = num_agents
        self.input_dim = num_agents * 11
        
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU()
        )
        self.gru = nn.GRU(64, num_agents * 64, num_layers=1, batch_first=True)
        
        self.actor = nn.Sequential(
            nn.Linear(num_agents * 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents + 1),  # outputs all actions
            nn.Sigmoid()
        )

    def forward(self, hist_seq: torch.Tensor) -> torch.Tensor:
        """
        hist_seq: (B, N, L, 11)
        """
        B, N, L, D = hist_seq.shape
        # Permute to (B, L, N, 11) and flatten N and 11
        x = hist_seq.permute(0, 2, 1, 3).reshape(B * L, N * D)
        
        embedded = self.embed(x)
        embedded = embedded.reshape(B, L, 64)
        
        gru_out, _ = self.gru(embedded)
        belief = gru_out[:, -1, :]  # (B, N*64)
        
        return self.actor(belief)
        
    def get_belief(self, hist_seq: torch.Tensor) -> torch.Tensor:
        B, N, L, D = hist_seq.shape
        x = hist_seq.permute(0, 2, 1, 3).reshape(B * L, N * D)
        
        embedded = self.embed(x)
        embedded = embedded.reshape(B, L, 64)
        
        gru_out, _ = self.gru(embedded)
        return gru_out[:, -1, :]
