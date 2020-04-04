from config import config
from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.Resize(config.resize),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(config.resize),
    transforms.ToTensor()
])
