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

    

def test_model_new(model, test_loader):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    if len(np.unique(all_labels)) == 2:
        all_scores = np.array(all_probs)[:, 1]
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.2f}")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    return model