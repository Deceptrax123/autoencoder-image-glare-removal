import torch 
import torchvision.transforms as T 
from torch.utils.data import DataLoader
from enhancement_dataset import EnhancementDataset,GlareDataset
from models.autoencoder.auto import Auto
from time import time 
from torch import nn
import torch.multiprocessing
import wandb 
from torch import mps 
from sklearn.model_selection import train_test_split
import gc 
import os 

def train_epoch():
    epoch_loss=0

    for step,(x_sample,y_sample) in enumerate(loader):
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)

        predictions=model(x_sample)
        model.zero_grad()

        #Compute loss
        loss=objective(predictions,y_sample)
        
        #perform backpropagation
        loss.backward()
        model_optimizer.step()
        
        epoch_loss+=loss.item()

        #Memory Management
        del x_sample 
        del y_sample
        del predictions 
        del loss
        mps.empty_cache()
        gc.collect(generation=2)
    
    loss=epoch_loss/train_steps
    return loss

def training_loop():

    for epoch in range(NUM_EPOCHS):
        model.train(True)
        loss=train_epoch()
        model.eval()

        with torch.no_grad():
            print("Epoch {epoch}".format(epoch=epoch+1))
            print("L1 Pixel Loss: {loss}".format(loss=loss))

            wandb.log({
                "L1 Pixel Loss":loss
            })

            #Checkpoints
            if((epoch+1)%10==0):
                path="./weights/autoencoder/run_2/model{epoch}.pth".format(epoch=epoch+1)
                torch.save(model.state_dict(),path)
                

if __name__=='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    labels=os.listdir("./errors/")

    params={
        'batch_size':8,
        'shuffle':True,
        'num_workers':0
    }

    #dataset
    dataset=GlareDataset(labels=labels)

    #logging
    wandb.init(
        project='autoencoder-image-enhancement',
        config={
            'architecture':'Autoencoder',
            'dataset':'Global Wheat Challenge'
        }
    )

    #data loaders
    loader=DataLoader(dataset,**params)

    #device
    if torch.backends.mps.is_available():
        device=torch.device("mps")
    else:   
        device=torch.device("cpu")

    #Hyperparameters
    LR=0.001
    NUM_EPOCHS=10000
    BETAS=(0.9,0.999)
    DECAY=0.001

    #Objective Function
    objective=nn.SmoothL1Loss()

    #Models
    model=Auto().to(device=device)
    model_optimizer=torch.optim.Adam(model.parameters(),lr=LR,betas=BETAS,weight_decay=DECAY)

    train_steps=(len(labels)+params['batch_size']-1)//params['batch_size']

    training_loop()