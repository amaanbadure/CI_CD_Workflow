# evaluate.py
import torch
from data_loader import load_data
from sacred import Experiment

ex = Experiment("CIFAR10_Workflow")

@ex.capture
def evaluate_model(model, device, _run):
    _, test_loader = load_data()
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    _run.log_scalar("test_accuracy", accuracy)
    return accuracy
