import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

df = pd.read_csv('Concrete_Data - Concrete_Data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class ConcreteMLP(nn.Module):
    def __init__(self, input_size):
        super(ConcreteMLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = ConcreteMLP(input_size=8)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor).item()
    r2 = r2_score(y_test_tensor, predictions)
    print(f'Test Loss (MSE): {test_loss:.4f}, R² Score: {r2:.4f}')

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions.numpy(), color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.title('Actual vs Predicted Plot')
plt.xlabel('Actual Compressive Strength (MPa)')
plt.ylabel('Predicted Compressive Strength (MPa)')
plt.legend()
plt.grid()
plt.show()

errors = y_test_tensor.numpy().flatten() - predictions.numpy().flatten()
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=10, color='skyblue', edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Prediction Error (MPa)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

samples = [
    [540, 0, 0, 162, 2.5, 1040, 676, 28],
    [332.5, 142.5, 0, 228, 0, 932, 594, 90]
]
samples_scaled = scaler.transform(samples)
samples_tensor = torch.FloatTensor(samples_scaled)

with torch.no_grad():
    sample_predictions = model(samples_tensor)
    print("Predicted Compressive Strengths:", sample_predictions.numpy())
