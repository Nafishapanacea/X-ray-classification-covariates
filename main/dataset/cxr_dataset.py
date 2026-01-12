import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.transforms import get_transforms
from utils.encoders import encode_view, encode_sex

class CXRMulitmodalDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transforms = get_transforms()
    
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        
        image = Image.open(row['path']).convert("RGB")
        image = self.transforms(image)
        
        
        view = encode_view(row['frontal/lateral'], row['AP/PA'])
        sex = encode_sex(row['sex'])
        
        
        view = torch.tensor(view, dtype=torch.long)
        sex = torch.tensor(sex, dtype=torch.long)
        
        
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)
        
        
        return image, view, sex, label