import torch
from torch.utils.data import DataLoader
from dataset.cxr_dataset import CXRMulitmodalDataset
from models.multimodal_cnn import MultimodalCNN
from config import *

def evaluate():
    dataset = CXRMulitmodalDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    model = MultimodalCNN().to(DEVICE)
    model.load_state_dict(torch.load("multimodal_cxr.pth"))
    model.eval()
    
    correct, total = 0, 0
    
    with torch.no_grad():
        for image, view, sex, label in loader:
            image = image.to(DEVICE)
            view = view.to(DEVICE)
            sex = sex.to(DEVICE)
            label = label.to(DEVICE)
                
            logits = model(image, view, sex)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            correct += (preds == label).sum().item()
            total += label.size(0)
    
    print(f"Accuracy: {correct / total:.4f}")