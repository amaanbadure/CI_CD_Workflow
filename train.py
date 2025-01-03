# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from data_loader import load_data
from sacred import Experiment

ex = Experiment("CIFAR10_Workflow")

@ex.capture
def train_model(_run, learning_rate, momentum, device, epochs, batch_size, num_workers):  # Add batch_size and num_workers
    train_loader, _ = load_data(batch_size=batch_size, num_workers=num_workers)
    
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        _run.log_scalar("training_loss", avg_loss, epoch)
    
    return model