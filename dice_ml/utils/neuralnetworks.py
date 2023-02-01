from torch import nn, sigmoid


class FFNetwork(nn.Module):
    def __init__(self, input_size, is_classifier=True):
        super(FFNetwork, self).__init__()
        self.is_classifier = is_classifier
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        out = sigmoid(out)
        if not self.is_classifier:
            out = 3 * out  # output between 0 and 3
        return out
