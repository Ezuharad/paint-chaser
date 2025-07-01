import torch
import torch.nn as nn
import matplotlib.pyplot as plt

agent = nn.Sequential(
    nn.Conv2d(1, 16, 10, 5),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.LeakyReLU(),
    nn.Linear(1920, 512),  # was 1028
    nn.LeakyReLU(),
    nn.Linear(512, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 4),
)

agent.load_state_dict(torch.load("../model_latest.pt"))

for module in agent:
    weight = module.weight
    for layer in weight:
        for channel in layer:
            plt.imshow(channel.unsqueeze(2).detach().cpu())
            plt.show()

    break
