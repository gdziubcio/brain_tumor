import torch
import torch.nn as nn
import torch.optim as optim
from src.models import SimpleCNN


def training_loop(
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    validate_loader, 
    num_epochs=10, 
    patience=None 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    trigger_counts = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()             # Zero the parameter gradients
            outputs = model(images)           # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                   # Backward pass
            optimizer.step()                  # Update parameters
        
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        avg_val_loss = validate_model(model, validate_loader, loss=loss)

        if patience: # turn on early stopping rule
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_counts = 0
            else: 
                trigger_counts += 1
                if trigger_counts > patience: 
                    print("Early stopping triggered!")
                    break

    return model
    
    
def validate_model(model, validate_loader, loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in validate_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()

    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    avg_val_loss = val_loss/len(validate_loader)
    return avg_val_loss

    


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return model 