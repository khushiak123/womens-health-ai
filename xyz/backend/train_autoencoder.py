import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Initialize Model
    autoencoder = Autoencoder(input_dim=7)
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    age = np.random.normal(45, 15, n_samples).reshape(-1, 1)
    bmi = np.random.normal(25, 5, n_samples).reshape(-1, 1)
    conditions = np.random.poisson(1, n_samples).reshape(-1, 1)
    activity = np.random.uniform(1, 10, n_samples).reshape(-1, 1)
    mental_health = np.random.normal(70, 15, n_samples).reshape(-1, 1)
    reproductive_health = np.random.normal(75, 15, n_samples).reshape(-1, 1)
    menopause = np.random.binomial(1, 0.3, n_samples).reshape(-1, 1)
    
    X_train = np.hstack([age, bmi, conditions, activity, 
                        mental_health, reproductive_health, menopause])
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_tensor = torch.FloatTensor(X_train_scaled)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    n_epochs = 200
    
    print("Training autoencoder...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = autoencoder(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    model_path = os.path.join(os.path.dirname(__file__), "autoencoder_model.pth")
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}")