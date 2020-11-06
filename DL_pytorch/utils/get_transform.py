import torchvision.transforms as transforms

def get_transform():
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    return transform
