from torch import nn


# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.block = nn.Sequential(nn.Linear(784, 128),
#                                    nn.ReLU(),
#                                    nn.Linear(128,128),
#                                    nn.ReLU(),
#                                    nn.Linear(128, 10))
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         return self.block(x)

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(32, 16, kernel_size=3, padding=1))

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(16 * 14 * 14, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 10))

    def forward(self, x):
        x = self.block(x)
        return self.classifier(x)
