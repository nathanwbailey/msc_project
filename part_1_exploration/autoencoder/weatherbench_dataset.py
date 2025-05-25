from torch.utils.data import Dataset
import torch
from create_forecast_data import create_forecast_data
import numpy as np

class WeatherBenchDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return x
    
class WeatherBenchDatasetWindow(Dataset):
    def __init__(self, data, context_length, target_length, stride=1):
        self.data = data
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride
        
    def __len__(self):
        return self.data.shape[0] - (self.context_length + self.target_length) // self.stride + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + self.context_length:idx + self.context_length + self.target_length]
        return x, y