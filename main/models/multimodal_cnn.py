import torch
import torch.nn as nn
from torchvision import models
from config import NUM_VIEW_TYPES, NUM_SEX_TYPES

class MultimodalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.densenet121(pretrained=True)
        num_feats = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        
        self.view_embedding = nn.Embedding(NUM_VIEW_TYPES, 8)
        self.sex_embedding = nn.Embedding(NUM_SEX_TYPES, 4)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(num_feats + 8 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, image, view, sex):
        img_features = self.backbone(image)
        view_emb = self.view_embedding(view)
        sex_emb = self.sex_embedding(sex)
        
        combined = torch.cat([img_features, view_emb, sex_emb], dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)