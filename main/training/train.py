import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset.cxr_dataset import CXRMulitmodalDataset
from models.multimodal_cnn import MultimodalCNN
from config import *

def train():
    dataset = CXRMulitmodalDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultimodalCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0   
        
        for image, view, sex, label in loader:
            image = image.to(DEVICE)
            view = view.to(DEVICE)
            sex = sex.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(image, view, sex)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "multimodal_cxr.pth")