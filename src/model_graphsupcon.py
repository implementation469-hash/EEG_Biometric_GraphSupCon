import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGraph(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_channels, num_channels))

    def forward(self, x):
        A = torch.softmax(self.A, dim=-1)
        x = torch.einsum("bct,oc->bot", x, A)
        return x, A

class GraphSupConEEGNet(nn.Module):
    def __init__(self, num_channels=64, num_classes=109, emb_dim=128):
        super().__init__()
        self.graph = LearnableGraph(num_channels)

        self.conv1 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(128, emb_dim)
        self.fc_cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_emb=False):
        x, A = self.graph(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x).squeeze(-1)
        emb = self.fc_emb(x)

        if return_emb:
            return emb, A

        logits = self.fc_cls(emb)
        return logits, emb, A
