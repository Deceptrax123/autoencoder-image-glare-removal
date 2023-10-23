import torch 
import torchvision.transforms as T 
import numpy as np
from models.autoencoder.auto import Auto
from PIL import Image
import matplotlib.pyplot as plt
import os 

def predict():
    device=torch.device("mps")
    model=Auto().to(device)

    model.eval()
    model.load_state_dict(torch.load("./weights/autoencoder/run_2/model3490.pth"))

    images=os.listdir('/Volumes/T7 Shield/Smudge/Datasets/Wheat_Head_detection/error/')
    transforms=T.Compose([T.Resize(size=(1024,1024)),T.ToTensor(),T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    for i in images:

        if '_' not in i:
            img=Image.open('/Volumes/T7 Shield/Smudge/Datasets/Wheat_Head_detection/error/'+i)

            img_tensor=transforms(img)
            img_tensor=img_tensor.to(device=device)

            img_tensor=img_tensor.view(1,img_tensor.size(0),img_tensor.size(1),img_tensor.size(2))
            
            prediction=model(img_tensor)

            prediction=prediction.to(device=torch.device("cpu"))
            prediction_np=prediction.detach().numpy()


            prediction_np=np.round((prediction_np+1)*255)//2

            prediction_np=prediction_np.astype(np.uint8)
            prediction_rgb=prediction_np.transpose(0,2,3,1)
            im=Image.fromarray(prediction_rgb[0])
            im.save("./autoencoder_enhanced/"+i)

predict()