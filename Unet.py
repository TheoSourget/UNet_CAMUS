"""
Base on tutorial :https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
Implementation Unet
"""

import torch
torch.manual_seed(1907)
import torch.nn as nn

from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor,RandomRotation
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split

from DownConvolution import DownConvolution
from LastConvolution import LastConvolution
from SimpleConvolution import SimpleConvolution
from UpConvolution import UpConvolution
from CamusEDImageDataset import CamusEDImageDataset


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


from EarlyStop import EarlyStopper
from tqdm import tqdm

from torchmetrics.functional import dice
import glob

class Unet(nn.Module):
    def __init__(self,input_channel,num_classes,light=False):
        super(Unet,self).__init__()
        if not light:        
            #PAPER UNET (31M parameters)
            #Encoder Part
            self.simpleConv = SimpleConvolution(input_channel,64)
            self.downBlock1 = DownConvolution(64,128)
            self.downBlock2 = DownConvolution(128,256)
            self.downBlock3 = DownConvolution(256,512)
    
            #Last level of Unet
            self.midmaxpool = nn.MaxPool2d(2,2)
            self.bridge = UpConvolution(512,1024)
    
            #Decoder Part
            self.upBlock1 = UpConvolution(1024,512)
            self.upBlock2 = UpConvolution(512,256)
            self.upBlock3 = UpConvolution(256,128)
            self.lastConv = LastConvolution(128,64,num_classes)
        else:
            #SIMPLIFIED UNET (3.7M parameters)
            self.simpleConv = SimpleConvolution(input_channel,22)
            self.downBlock1 = DownConvolution(22,44)
            self.downBlock2 = DownConvolution(44,88)
            self.downBlock3 = DownConvolution(88,176)
    
            self.midmaxpool = nn.MaxPool2d(2,2)
            self.bridge = UpConvolution(176,352)
    
            self.upBlock1 = UpConvolution(352,176)
            self.upBlock2 = UpConvolution(176,88)
            self.upBlock3 = UpConvolution(88,44)
            self.lastConv = LastConvolution(44,22,num_classes)

    def forward(self,x):
        x_1 = self.simpleConv(x)
        x_2 = self.downBlock1(x_1)

        x_3 = self.downBlock2(x_2)
        x_4 = self.downBlock3(x_3)

        x_5 = self.midmaxpool(x_4)

        x_6 = self.bridge(x_5)

        x_4_6 = torch.cat((x_4,x_6),1)

        x_7 = self.upBlock1(x_4_6)
         
        x_3_7 = torch.cat((x_3,x_7),1)

        x_8 = self.upBlock2(x_3_7)

        x_2_8 = torch.cat((x_2,x_8),1)
        x_9 = self.upBlock3(x_2_8)

        x_1_9 = torch.concat((x_1,x_9),1)
        out = self.lastConv(x_1_9)

        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NB_EPOCHS = 50
    NBSAMPLES = len(glob.glob("./data/training/**/*_ED.mhd"))
    VALID_SIZE = 2

    #Use a "light" version if True (3.7M params) or the paper version if False (31M params)
    lightUnet = False
    
    #Load camus dataset
    train_data =CamusEDImageDataset(
        transform=Compose([ToPILImage(),Resize((256,256)),RandomRotation(10),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((256,256)),RandomRotation(10),PILToTensor()]),
    )

    valid_data =CamusEDImageDataset(
        transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((256,256)),PILToTensor()]),
    )

    #Split with validation set
    train_indices, val_indices = train_test_split(np.arange(0,NBSAMPLES,1),test_size=VALID_SIZE,random_state=1907)

    train_data = torch.utils.data.Subset(train_data,train_indices)
    valid_data =torch.utils.data.Subset(valid_data,val_indices)
    
    #Turn the dataset into DataLoader
    train_dataloader = DataLoader(train_data, batch_size=5)
    valid_dataloader = DataLoader(valid_data, batch_size=5)
    
    
    net = Unet(1,4,light=lightUnet).to(device)

    optimizer = optim.Adam(net.parameters(),lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    criterion.requires_grad = True
    
    lossEvolve = []
    valEvolve = []
    diceEvolve = []
    es = EarlyStopper(3,0.1)

    #For animation
    imgs = []
    for epoch in tqdm(range(NB_EPOCHS)):  # loop over the dataset multiple times
        net.train()
        print("################# EPOCH:",epoch+1,"#################")

        #Train
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        #Validation
        net.eval()
        val_loss = 0.0
        dice_curr = 0.0
        with torch.no_grad():
            for j, data in enumerate(valid_dataloader, 0):
                inputs, labels = data
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                labels = labels.squeeze(1)
                loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
                #loss.requires_grad = True
                val_loss += loss.item()
                dice_curr += dice(outputs,labels,average="micro",ignore_index=0)

            #For animation
            inputs, labels = valid_dataloader.dataset[0]
            if epoch == 0:
                baseImage = inputs[0]
                baseImage = baseImage.detach().cpu().numpy()

            inputs = inputs.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            outputs = net(inputs)
            outputs = torch.softmax(outputs,1)
            pred = torch.Tensor(torch.argmax(outputs,1).float())
            imgs.append(pred.detach().cpu().numpy()[0])

        lossEvolve.append(train_loss/(i+1))
        valEvolve.append(val_loss/(j+1))
        diceEvolve.append(dice_curr.cpu()/(j+1))
        print("Training Loss: %f \tValid Loss: %f \tDice: %f"%(train_loss/(i+1),val_loss/(j+1),dice_curr/(j+1)))
        
        #Early stopping
        if es.early_stop(val_loss/(j+1)):
            break
        if val_loss/(j+1) == min(valEvolve):
            torch.save(net.state_dict(),'./weights/Unet.pt')


    print('Finished Training')
    plt.figure(figsize=(5,5))
    plt.plot(lossEvolve,label="Train set loss")
    plt.plot(valEvolve,label="Validation set loss")
    plt.title("Evolution of loss for validation and train dataset")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(diceEvolve)
    plt.title("Evolution of Dice metric on valdiation set")
    plt.show()


    palette = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])
    def updateSeg(i,imgBase,segEvolve):
        plt.clf()
        plt.axis("off")
        plt.title(f"Evolution of segmentation with Unet: Epochs {i+1}")
        seg = segEvolve[i]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        plt.imshow(imgBase,cmap="gray")
        plt.imshow(color_seg,alpha=0.3)
    
    #Animation
    frames = [] # for storing the generated images
    fig = plt.figure()
    plt.axis("off")
    plt.title(f"Evolution of segmentation with Unet during {NB_EPOCHS} Epochs")
    ani = animation.FuncAnimation(fig, updateSeg,frames=len(imgs), interval=1000,repeat_delay=300,fargs=(baseImage,imgs))
    writergif = animation.PillowWriter(fps=30) 
    ani.save('SimplifiedUnet.gif', writer=writergif)
    plt.show()