# Importing the libraries

from random import shuffle
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils import visualize_augmentations, seed_everything, plot_data
import pandas as pd
import random
from torchvision import models
# from model import Net



device = 'cuda' if torch.cuda.is_available() else 'cpu'


seed_everything(42)

import os
from torch.utils.data import Dataset
import glob
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from pathlib import Path

class CatsnDogsDataset(Dataset):
    dataset_folder_name = 'data'
    
    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    

    def __init__(self, root, train=True, transform=None):
        # self.dataset_folder_name = dataset_folder_name
        self.samples = []
        self.EXTENSION = 'jpg'
        self.root = root
        self.transform = transform

        self.split_dir = 'train' if train else 'validation'
        
        self.split_dir = os.path.join(
            self.root, self.dataset_folder_name, self.split_dir)
        
        _, self.classes_dict = self.find_classes(self.split_dir)
    
        self.image_paths = sorted(glob.iglob(os.path.join(
            self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.image_paths = random.sample(self.image_paths, len(self.image_paths))
    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        filepath = self.image_paths[index]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = Path(filepath).parent.absolute().name

        if self.transform:  
            '''Performing augmentation'''   
            image = self.transform(image=image)

        return image, self.classes_dict[target]

def get_transforms(mode='train'):
    if mode == 'train':
        print('Performing training augmentation')
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
                A.RandomCrop(height=128, width=128),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        return train_transform
    else:
        print('Performing val augmentation')

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.CenterCrop(height=128, width=128),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
        return val_transform


def train(model, device, train_dataloader, optimizer, loss_fn):

    correct = 0
    processed = 0
    train_acc = []
    train_losses = []

    model.train()
    pbar = tqdm(train_dataloader)

    for batch_idx, (data, target) in enumerate(pbar):
        # print('weqwr', data, target)
        data, target = data['image'].to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        pred = y_pred.argmax(dim=1, keepdim=True)

        # pred = pred.squeeze()

        # print(type(target), target.shape)
        # print(type(pred), pred.shape)

        target_oh = torch.nn.functional.one_hot(target).float()

        loss_fn = torch.nn.BCEWithLogitsLoss()
        # print('target_oh == ', target_oh)
        try:

            loss = loss_fn(y_pred, target_oh)
        except Exception as e:
            print('dfwfe', target, target_oh, torch.nn.functional.one_hot(target[0]).float())

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()


        correct += torch.sum(pred.squeeze() == target).sum()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return sum(train_acc)/len(train_acc), sum(train_losses)/len(train_losses)

def test(model, device, val_dataloader, loss_fn):

    test_loss = 0
    correct = 0
    cats_correct = 0
    cats_total = 0
    dogs_total = 0
    dogs_correct = 0

    test_acc = []
    test_losses = []

    model.eval()
    
    with torch.no_grad():

        for data, target in val_dataloader:
            data, target = data['image'].to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for label, predict in zip(target, pred):
                if(label == 0):
                    cats_total += 1
                    
                    if(label == predict[0]):
                        cats_correct += 1
                else:
                    dogs_total += 1
                    if(label == predict[0]):
                        dogs_correct += 1 

            loss_fn = torch.nn.BCEWithLogitsLoss()  
            target_oh = torch.nn.functional.one_hot(target).float()
            
            test_loss += loss_fn(output, target_oh)

            correct += torch.sum(pred.squeeze() == target).sum()

    test_loss /= len(val_dataloader.dataset)
    test_losses.append(test_loss)

    # Class wise accuracy
    dogs_accuracy = dogs_correct/dogs_total*100
    cats_accuracy = cats_correct/cats_total*100   
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%, Cats_Accuracy {:.2f}%, Dogs_Accuracy {:.2f}%)\n'.format(
        test_loss, correct, len(val_dataloader.dataset),
        100. * correct / len(val_dataloader.dataset), cats_accuracy, dogs_accuracy))
 
    return test_loss, 100. * correct / len(val_dataloader.dataset), dogs_accuracy, cats_accuracy

def run(EPOCHS = 2, BATCH_SIZE=256):

    use_pretrained = True
    model = models.vgg16(pretrained=use_pretrained)

    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

    # Specify The Layers for updating
    params_to_update = []

    update_params_name = ['classifier.6.weight', 'classifier.6.bias']

    for name, param in model.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # model = Net().to(device)

    train_dataset = CatsnDogsDataset(root=os.getcwd(), transform=get_transforms())
    val_dataset = CatsnDogsDataset(root=os.getcwd(), train=False, transform=get_transforms(mode='val'))

    visualize_augmentations(train_dataset)
    visualize_augmentations(val_dataset)

    dataloader_args = dict(shuffle=True, num_workers=4, batch_size=BATCH_SIZE, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset,**dataloader_args)
    val_dataloader = DataLoader(val_dataset, **dataloader_args)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.06)


    loss = torch.nn.BCELoss()


    df = pd.DataFrame(
        columns = ['Epoch' , 'Train_loss', 'Train_acc' , 'Test_loss', 'Test_acc', 'Cats_Accuracy', 'Dogs_Accuracy'])

    for epoch in range(EPOCHS):
        print('EPOCH - ', epoch)

        train_epoch_loss, train_epoch_acc = train(model, device, train_dataloader, optimizer, loss)
        scheduler.step()
        test_epoch_loss, test_epoch_acc, test_cats_acc, test_dogs_acc = test(model, device, val_dataloader, loss)
        series_data = pd.Series([int(epoch), train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc, test_cats_acc, test_dogs_acc], index = df.columns)   
        df = df.append(series_data, ignore_index=True)

    df.to_csv('metrics.csv')
    torch.save(model.state_dict(), f'model_{EPOCHS}ep_{int(test_epoch_acc)}.pt')
    return model
            


if __name__ == '__main__':
    run()


