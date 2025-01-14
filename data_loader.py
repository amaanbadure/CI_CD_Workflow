# data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sacred import Experiment
import platform

ex = Experiment("CIFAR10_Workflow")

@ex.capture
def load_data(batch_size=64, num_workers=None):
    # Set num_workers based on platform
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 2
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                  download=True, transform=transform)
    
    # On Windows, we'll use num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=False if num_workers == 0 else True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=False if num_workers == 0 else True
    )
    
    return train_loader, test_loader