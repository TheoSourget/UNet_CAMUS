"""
Script to test the U-Net model on the test set given by the challenge
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.manual_seed(1907)
from CamusEDImageDataset import CamusEDImageDataset

from Unet import Unet
from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.functional import dice
from tqdm import tqdm
from datasets import load_metric

from torchmetrics.functional import dice

#If True, will display a visual exemple of the segmentation instead of testing all the test set to compute metric
VISUAL = False
metric = load_metric("mean_iou")

net = Unet(1,4,light=False)
net.load_state_dict(torch.load('./weights/Unet.pt'))

net.eval()

#Load dataset
test_data =CamusEDImageDataset(
    transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
    target_transform=Compose([ToPILImage(),Resize((256,256)),PILToTensor()]),
    test=True
)

test_dataloader = DataLoader(test_data, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if VISUAL :
    idxImgTest = 0
    inputs, labels = test_dataloader.dataset[idxImgTest]
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
    outputs = net(inputs)
    outputs = torch.softmax(outputs,1)
    new_labels = torch.squeeze(labels)
    pred = torch.Tensor(torch.argmax(outputs,1).float())
    output = outputs[0].detach().cpu().numpy()
    
    plt.figure(figsize=(20,20))
    plt.subplot(131)
    plt.title("Echography image")
    plt.axis("off")
    plt.imshow(inputs[0][0],cmap="gray")
    
    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(new_labels.detach().cpu().numpy())
    plt.axis("off")
    
    plt.subplot(133)
    plt.title("Model Segmentation")
    plt.imshow(pred.detach().cpu().numpy()[0],vmax=3,vmin=0)
    plt.axis("off")

    plt.show()
else:
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    dices = {i:[] for i in range(4)}
    
    with torch.no_grad():
        val_loss = 0.0
        for j, data in enumerate(tqdm(test_dataloader, 0)):
            inputs, labels = data

            outputs = net(inputs.to(device))
            labels = labels.squeeze(1)
            loss = criterion(outputs.to(device), labels.type(torch.LongTensor).to(device))
            metric.add_batch(predictions=outputs.argmax(dim=1).detach().cpu().numpy(), references=labels.type(torch.LongTensor).detach().cpu().numpy())
            
            dice_metric = dice(outputs.detach().cpu(),labels.detach().cpu(),average = None,num_classes=4,ignore_index=0)
            for i in range(len(dice_metric)):
                dices[i].append(dice_metric[i].item())
            
            val_loss += loss.item()
            
        metrics = metric.compute(
            num_labels=4, 
            ignore_index=0,
            reduce_labels=False,
        )
        print("TEST METRICS")
        print("Mean_iou:", np.mean(metrics["per_category_iou"][1:])) #Remove first value because it's 0 and not nan 
        print("Mean accuracy:", metrics["mean_accuracy"])
        print("iou per category:", metrics["per_category_iou"])
        print("Accuracy per category:", metrics["per_category_accuracy"])
        
        print('DICES:')
        for i in range(1,4):
            print("Class",i)
            print(f"Mean: {np.mean(dices[i])}\t+/-{np.std(dices[i])}")
        
        