import torch.nn as nn
import torch


class DiffusionStateLinear(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.feature_dim=hidden_dim
        self.module = nn.Sequential(
            nn.Linear(8, 128),
            nn.GELU()
        )
    def forward(self, state):
        return self.module(state)

class DiffusionStateMLP(nn.Module):
    def __init__(self, joint_dim=8, embed_dim=64, output_dim=128):
        super().__init__()
        self.feature_dim=output_dim
        self.joint_embeddings = nn.Parameter(torch.randn(joint_dim, embed_dim))  # Learned embedding per joint
        self.module = nn.Sequential(
            nn.Linear(joint_dim*embed_dim, output_dim*2),
            nn.GELU(),
            nn.Linear(output_dim*2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, joint_values):
        joint_values = joint_values.unsqueeze(-1)                    # (B, S, 8, 1)
        joint_embeds = self.joint_embeddings.unsqueeze(0)            # (1, S, 8, 64)
        joint_embeds_scaled = joint_values * joint_embeds            # (B, S, 8, 64)
        joint_embeds_flat = joint_embeds_scaled.flatten(start_dim=2) # (B, S, 512)
        return self.module(joint_embeds_flat)                          # (B, S, 128)

class RIBSStateEncoder(nn.Module):
    def __init__(self, joint_dim=8, embed_dim=64, output_dim=128):
        super().__init__()
        self.feature_dim = output_dim
        self.joint_dim = joint_dim
        self.embed_dim = embed_dim
        self.joint_embeddings = nn.Parameter(torch.randn(joint_dim, embed_dim))  # Learned embedding per joint

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim*joint_dim, output_dim*2),
            nn.GELU(),
            nn.Linear(output_dim*2, output_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(output_dim, output_dim)
        self.logvar_head = nn.Linear(output_dim, output_dim)

    def forward(self, joint_values):
        """
        joint_values: (B, S, 8) - continuous joint angles
        """
        # (8, 128) * (B, 8, 1) → (B, 8, 128)
        joint_values = joint_values.unsqueeze(-1)                    # (B, 8, 1)
        joint_embeds = self.joint_embeddings.unsqueeze(0)            # (1, 8, 64)
        joint_embeds_scaled = joint_values * joint_embeds            # (B, 8, 64)
        joint_embeds_flat = joint_embeds_scaled.flatten(start_dim=2) # (B, 512)
        h = self.encoder(joint_embeds_flat)                          # (B, 128)
        mu = self.mu_head(h)                                         # (B, 128)
        logvar = self.logvar_head(h)                                 # (B, 128)

        return mu, logvar