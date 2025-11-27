import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleNet


def generate_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """
    Generate dummy data for training.
    Returns: (X, y) where X is (num_samples, input_size) and y is (num_samples,)
    """
    # Generate random input data: (num_samples, input_size)
    X = torch.randn(num_samples, input_size)
    # Generate random labels: (num_samples,)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train():
    # Set device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    input_size = 784
    hidden_size = 128
    num_classes = 10
    
    # Generate dummy dataset
    print("Generating dummy dataset...")
    X_train, y_train = generate_dummy_data(num_samples=1000, input_size=input_size, num_classes=num_classes)
    X_val, y_val = generate_dummy_data(num_samples=200, input_size=input_size, num_classes=num_classes)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = SimpleNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data = data.to(device)  # (batch_size, 784)
            targets = targets.to(device)  # (batch_size,)
            
            # Forward pass
            outputs = model(data)  # (batch_size, 10)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)  # (batch_size, 784)
                targets = targets.to(device)  # (batch_size,)
                
                outputs = model(data)  # (batch_size, 10)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")
    
    print("\nTraining completed!")
    
    # Save model
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    train()

