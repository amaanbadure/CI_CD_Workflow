# data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sacred import Experiment

ex = Experiment("CIFAR10_Workflow")

@ex.capture
def load_data(batch_size=64, num_workers=2):  # Add default values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader