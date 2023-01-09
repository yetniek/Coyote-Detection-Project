# importing all the modules
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CNNclassification(torch.nn.Module):
    def __init__(self):
        # Add layers
        super(CNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(40, 10, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(10, 100, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(200, 300, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.fc_layer = nn.Sequential(
            nn.Linear(3000, 2)  # fully connected layer(ouput layer)
        )

    def forward(self, x):
        x = self.layer1(x)  # 1-layer
        x = self.layer2(x)  # 2-layer
        x = self.layer3(x)  # 3-layer
        x = self.layer4(x)  # 4-layer
        x = torch.flatten(x, start_dim=1)  # Convert N-dimensional array to 1-dimensional array 

        out = self.fc_layer(x)
        return out

# Define Dataset 
class CustomDataset(Dataset):
    def __init__(self, file_name, train_mode=True, transforms=None):  
        self.train_mode = train_mode
        self.transforms = transforms
        self.file_name = file_name

    # return length
    def __len__(self): 
        return 1

    def __getitem__(self, index): 
        audio, sr = librosa.load(self.file_name, sr=16000, duration=5.0)
        audio = np.array(audio)
        audio = audio.reshape(audio.shape[0], audio.shape[1], 1)  

        label = 0

        if self.transforms is not None:  # Transform check
            audio = self.transforms(audio)

        return audio, label  # Size check  

def predict(file_name):
    best_model = CNNclassification().to(device)
    best_model.load_state_dict(torch.load('./MFCC_16000_best_model.pth', map_location=torch.device('cpu')))
    vali_dataset = CustomDataset(file_name, train_mode = False, transforms = None)
    vali_loader = DataLoader(vali_dataset, batch_size = 1, shuffle=False, num_workers=0)

    best_model.eval()
    with torch.no_grad():  # Detach from the current graph, set to False
        for wav, label in iter(vali_loader):
            wav, label = wav.to(device), label.to(device)
            logit = best_model(wav)
            pred = logit.argmax(dim=1, keepdim=False)  # Extract the highest value among 10 classes as the predicted label
            return pred.item()