import torch
from torch import manual_seed, cuda
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import random


def seed_everything(seed: int) -> None:
    '''Seeds the Code so that we get predictable outputs'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)
    if torch.cuda.is_available():
        cuda.manual_seed(SEED)


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):

    dataset = copy.deepcopy(dataset)
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image['image'].permute(1, 2, 0))
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_data(train_losses,train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2, figsize=(15,10))
    axs[0,0].plot(train_losses)
    axs[0,0].set_title('train_losses')
    axs[0,1].plot(train_acc)
    axs[0,1].set_title('Training Accuracy')
    axs[1,0].plot(test_losses)
    axs[1,0].set_title('Test Losses')
    axs[1,1].plot(test_acc)
    axs[1,1].set_title('Test Accuracy')