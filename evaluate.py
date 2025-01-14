#evaluate.py
import torch
from data_loader import load_data
from sacred import Experiment

ex = Experiment("CIFAR10_Workflow")

@ex.capture
def evaluate_model(model, _run, device, batch_size, num_workers):
    _, test_loader = load_data(batch_size=batch_size, num_workers=num_workers)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Store results in the format expected by the checker
    _run.info['test_accuracy'] = accuracy
    
    return accuracy