import torch 
import torchvision
import torchvision.transforms as T
from PIL import Image 
import numpy as np

class EnhancementDataset(torch.utils.data.Dataset):
    def __init__(self,labels):
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):

        label=self.labels[index]
        
        image_x=Image.open("./errors/"+label)
        image_y=Image.open("./enhanced_errors/"+label)

        image_xnp=np.array(image_x).astype(np.float32)
        image_ynp=np.array(image_y).astype(np.float32)
        
        transforms=T.Compose([T.Resize(size=(1024,1024)),T.ToTensor(),T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        image_xtensor=transforms(image_xnp)
        image_ytensor=transforms(image_ynp)

        return image_xtensor,image_ytensor