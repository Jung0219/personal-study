import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Improved Model with BatchNorm and Dropout


class ClassificationModel(nn.Module):
    def __init__(self, n_features, n_hidden_layers1, n_hidden_layers2, n_labels):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_layers1)
        self.bn1 = nn.BatchNorm1d(n_hidden_layers1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(n_hidden_layers1, n_hidden_layers2)
        self.bn2 = nn.BatchNorm1d(n_hidden_layers2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(n_hidden_layers2, n_labels)

    def forward(self, data):
        data = F.relu(self.bn1(self.fc1(data)))
        data = self.dropout1(data)
        data = F.relu(self.bn2(self.fc2(data)))
        data = self.dropout2(data)
        data = self.fc3(data)  # Output logits
        return data

# Dataset class remains unchanged


class WineDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = pd.DataFrame(x).values
        self.y = pd.DataFrame(y).values
        self.n_features = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        features, labels = self.x[index], self.y[index]
        if self.transform:
            features = self.transform(features)
        return features, torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return self.n_features


class ToTensor:
    def __call__(self, data):
        return torch.tensor(data, dtype=torch.float32)


# Load data and apply feature scaling
data = load_wine()
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    scaler.fit_transform(data.data), data.target, test_size=0.8, random_state=42
)

train_data = WineDataset(X_train, y_train, transform=ToTensor())
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

# Instantiate model, loss function, and optimizer
model = ClassificationModel(13, 32, 16, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0027)

# Training loop with increased epochs
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_dataloader:
        # Forward pass
        predictions = model(features)
        loss = criterion(predictions, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

# Test evaluation
test_data = WineDataset(X_test, y_test, ToTensor())
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_dataloader:
        predictions = model(features)
        # print(f"prediction = {torch.ceil(torch.max(predictions))}, value = {labels}")
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
