from torchvision import transforms

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        transforms.Normalize(mean=[0.502]*3, std=[0.284]*3)
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        transforms.Normalize(mean=[0.502]*3, std=[0.284]*3)
    ])

    # Mean: 0.502022, Std: 0.284344