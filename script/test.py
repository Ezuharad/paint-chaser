import torchvision
import torch
import matplotlib.pyplot as plt

image = torch.load("block.tn")

print(image)
print(image.shape)

plt.imshow(image)
plt.show()

