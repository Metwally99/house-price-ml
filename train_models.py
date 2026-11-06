import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = os.environ.get("DATASET_PATH", os.path.join(os.path.dirname(__file__), "data", "houses.csv"))
MODEL_SAVE_PATH = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

def load_and_preprocess_data():
    print("Loading data...")
    # Read CSV with inferred columns based on previous analysis
    # Columns: serial, date, age, distance, stores, latitude, longitude, price
    df = pd.read_csv(DATA_PATH)
    
    # Assign column names if not present (assuming the file has no header based on previous `cat` output)
    # The user's notebook output showed these columns for 'data.csv', assuming 'Data_Set.csv' is the same format.
    # We saw '0,2009,21,9,6,84,121,14264' in existing file, which matches the notebook's head().
    df.columns = ['serial', 'date', 'age', 'distance', 'stores', 'latitude', 'longitude', 'price']
    
    # Drop serial as it's an ID
    df = df.drop('serial', axis=1)
    
    X = df[['date', 'age', 'distance', 'stores', 'latitude', 'longitude']].values
    y = df['price'].values.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"Data loaded. Train shape: {X_train_tensor.shape}, Test shape: {X_test_tensor.shape}")
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler

# Define Models
class ModelA(nn.Module): # Linear Regression (Baseline)
    def __init__(self, input_dim):
        super(ModelA, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

class ModelB(nn.Module): # Shallow NN
    def __init__(self, input_dim):
        super(ModelB, self).__init__()
        self.hidden = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class ModelC(nn.Module): # Deep NN
    def __init__(self, input_dim):
        super(ModelC, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        return self.output(x)

class ModelD(nn.Module): # Wider NN
    def __init__(self, input_dim):
        super(ModelD, self).__init__()
        self.hidden = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class ModelE(nn.Module): # Deeper NN (3 Hidden Layers)
    def __init__(self, input_dim):
        super(ModelE, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.hidden3(x)
        x = self.relu3(x)
        return self.output(x)

def manual_train(model, optimizer, X_train, y_train, epochs=1000):
    model.train()
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        # 1. Forward pass
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        
        # 2. Manual Gradient Calculation Steps
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients (Chain Rule)
        optimizer.step()      # Update weights
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y_test)
    return loss.item()

def main():
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data()
    input_dim = X_train.shape[1]
    
    results = {}
    
    # Train Model A
    print("\nTraining Model A (Linear Regression)...")
    model_a = ModelA(input_dim)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
    train_a = manual_train(model_a, opt_a, X_train, y_train)
    loss_a = evaluate(train_a, X_test, y_test)
    results['Model A'] = loss_a
    torch.save(model_a.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model_a.pth'))
    
    # Train Model B
    print("\nTraining Model B (Shallow NN)...")
    model_b = ModelB(input_dim)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    train_b = manual_train(model_b, opt_b, X_train, y_train)
    loss_b = evaluate(train_b, X_test, y_test)
    results['Model B'] = loss_b
    torch.save(model_b.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model_b.pth'))

    # Train Model C
    print("\nTraining Model C (Deep NN)...")
    model_c = ModelC(input_dim)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=0.01) # Slightly lower LR for deep net logic often involves tuning, but 0.01 is standard
    train_c = manual_train(model_c, opt_c, X_train, y_train)
    loss_c = evaluate(train_c, X_test, y_test)
    results['Model C'] = loss_c
    torch.save(model_c.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model_c.pth'))

    # Train Model D
    print("\nTraining Model D (Wider NN)...")
    model_d = ModelD(input_dim)
    opt_d = torch.optim.Adam(model_d.parameters(), lr=0.01)
    train_d = manual_train(model_d, opt_d, X_train, y_train)
    loss_d = evaluate(train_d, X_test, y_test)
    results['Model D'] = loss_d
    torch.save(model_d.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model_d.pth'))

    # Train Model E
    print("\nTraining Model E (Deeper NN)...")
    model_e = ModelE(input_dim)
    opt_e = torch.optim.Adam(model_e.parameters(), lr=0.005) # Lower LR for deeper network
    train_e = manual_train(model_e, opt_e, X_train, y_train)
    loss_e = evaluate(train_e, X_test, y_test)
    results['Model E'] = loss_e
    torch.save(model_e.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model_e.pth'))
    
    # Save Scaler for Web App
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'))
    
    print("\n--- Final Results (Test MSE) ---")
    print(f"{'Model':<20} | {'Test MSE':<15}")
    print("-" * 35)
    for model_name, loss in results.items():
        print(f"{model_name:<20} | {loss:.4f}")

if __name__ == "__main__":
    main()
