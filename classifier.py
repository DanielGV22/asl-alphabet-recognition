import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=63, num_classes=26, hidden=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

@torch.inference_mode()
def predict(model, feats_1d, device):
    x = torch.from_numpy(feats_1d).to(device).unsqueeze(0)  # (1,63)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    prob = float(probs[idx].item())
    return idx, prob
