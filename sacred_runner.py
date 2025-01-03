# sacred_runner.py
from sacred import Experiment
from sacred.observers import FileStorageObserver
from train import train_model
from evaluate import evaluate_model
import torch

# Create Sacred experiment
ex = Experiment("CIFAR10_Workflow")
ex.observers.append(FileStorageObserver.create("experiment_logs"))

@ex.config
def config():
    # Dataset parameters
    batch_size = 64
    num_workers = 2
    
    # Model parameters
    input_size = 3 * 32 * 32
    num_classes = 10
    
    # Training parameters
    epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

@ex.main
def main(_run, learning_rate, momentum, device, epochs, batch_size, num_workers):
    # Run training
    model = train_model(
        _run=_run,
        learning_rate=learning_rate,
        momentum=momentum,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Run evaluation
    # Run evaluation with device parameter
    accuracy = evaluate_model(
        model=model, 
        _run=_run,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return accuracy

if __name__ == "__main__":
    ex.run()