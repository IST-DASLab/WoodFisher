import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(5,3)

    def forward(self, input_tensor):
        return self.fc(input_tensor)