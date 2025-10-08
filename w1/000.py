import torch
from torch import nn, optim
from model import Model # Assuming this is a simple linear model
import matplotlib.pyplot as plt
import os
import numpy as np

# Create a dummy CSV file for demonstration if it doesn't exist
csv_data = """distance_km,fare_twd
1.0,75
2.5,100
5.0,160
7.5,210
10.0,270
15.0,380
"""
csv_file_path = 'taxi_data.csv'
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w') as f:
        f.write(csv_data)

# --- CSV Data Loading and Preprocessing ---
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
x = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1) # Kilometers
y = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1) # Fares

# Check if data is loaded correctly
print(f'Loaded {x.shape[0]} data points.')
print(f'First 5 kilometers: {x[:5].squeeze().tolist()}')
print(f'First 5 fares: {y[:5].squeeze().tolist()}')

# --- Model Initialization and Training ---
torch.manual_seed(0)

net = Model()
opt = optim.SGD(net.parameters(), lr=0.01) # Reduced learning rate for stability
loss_f = nn.MSELoss()
loss_hist = []

num_epochs = 1000 # Increased epochs for better convergence
print("\nStarting training...")
for epoch in range(num_epochs):
    y_hat = net(x)
    loss = loss_f(y, y_hat)
    loss.backward()
    opt.step()
    opt.zero_grad()
    loss_hist.append(float(loss.item()))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Plotting and Saving Results ---
plt.figure()
plt.plot(range(len(loss_hist)), loss_hist, marker='o', markersize=2, linestyle='--')
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, 'loss.png')
plt.savefig(loss_path)
plt.close()
print(f'\nLoss plot saved to {loss_path}')

# --- Model Evaluation and Prediction Saving ---
net.eval()
with torch.no_grad():
    y_hat_pred = net(x)

arr = np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat_pred.detach().cpu().numpy()
])

pred_csv_path = os.path.join(OUT_DIR, 'predictions.csv')
np.savetxt(pred_csv_path, arr, delimiter=',', header='distance,fare,predicted_fare', comments='')
print(f'Predictions saved to {pred_csv_path}')

# --- Visualization of Model Fit ---
plt.figure()
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 'o', alpha=0.6, label='Actual Data (distance, fare)')

# Sort data for a clean connected line plot
idx = x.squeeze(1).argsort()
x_sorted = x[idx]
y_hat_sorted = y_hat_pred[idx]

plt.plot(x_sorted.detach().cpu().numpy(), y_hat_sorted.detach().cpu().numpy(), '-', linewidth=2, label='Model Fit Line')

plt.xlabel('Distance (km)')
plt.ylabel('Fare (TWD)')
plt.title('Model Prediction on Training Data')
plt.legend()
plt.tight_layout()
fit_plot_path = os.path.join(OUT_DIR, 'model_fit.png')
plt.savefig(fit_plot_path, dpi=150)
plt.show()

# --- Saving the trained model ---
model_save_path = os.path.join(OUT_DIR, 'trained_model.pth')
torch.save(net.state_dict(), model_save_path)
print(f'Trained model saved to {model_save_path}')

# --- Example of using the trained model ---
print("\n--- Example prediction ---")
new_distance = torch.tensor([[8.5]], dtype=torch.float32)
with torch.no_grad():
    predicted_fare = net(new_distance)
print(f'For a distance of {new_distance.item()} km, the estimated fare is {predicted_fare.item():.2f} TWD.')