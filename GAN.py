import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import os


def image_loader(image_name, imgsize, device):
    loader = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.CenterCrop(imgsize),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def transfer(img_path, style, imgsize=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('./models_pretrained/style_' + style + '.pth').to(device)
    img = image_loader(img_path, imgsize, device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model(img)


def draw_img(img):
    plt.imshow(np.rollaxis(img.add(1).div(2).cpu().detach()[0].numpy(), 0, 3))
    plt.show()


if __name__ == '__main__':
    pass
