import os
import numpy as np
import torch
from PIL import Image

class XXXDataset(object):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):
        return image, target

    def __len__(self):
        return len(self.imgs)


# test XXDataset
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
if __name__ == '__main__':

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(np.img, (1,2,0)))
        plt.show()

    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    dataset = XXXDataset(root='./data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(labels)
