import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

data = []

def preprocess_data():
    """
    Load and preprocess data from CSV file.
    """
    global data

    data = pd.read_csv('./data/demand_temperature.csv')
    data['Time'] = pd.to_datetime(data['Time'])
    data['Time'] = data['Time'].apply(lambda x: x.hour)
    X = data[['Time', 'Temperature']].values
    y = data[['Demand']].values
    return X, y

def create_data_loader(X, y, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader object from input and target data.
    """
    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def create_model():
    """
    Create a PyTorch model with one hidden layer and ReLU activations.
    """
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        # nn.Dropout(0.8),

        nn.Linear(10, 10),
        nn.ReLU(),
        # nn.Dropout(0.8),
        
        # nn.Linear(10, 50),
        # nn.ReLU(),
        
        # nn.Linear(50, 10),
        # nn.ReLU(),
        
        nn.Linear(10, 1),
        # nn.Sigmoid(),
    )
    return model

loss_history = []

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    """
    Train a PyTorch model using the given data, optimizer, and loss function.
    """
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # print('X:', X, X.shape)
            output = model(X)
            # print('y:', y)
            # print('output:', output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            # print('Batch loss:', loss.item())
        train_loss /= len(train_loader.dataset)

        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item() * X.size(0)
            val_loss /= len(val_loader.dataset)

        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        loss_history.append(train_loss)


if __name__ == '__main__':
    # Preprocess data
    X, y = preprocess_data()

    # Split data into train and val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create data loaders
    train_loader = create_data_loader(X_train, y_train, batch_size=5) #len(X_train))
    val_loader = create_data_loader(X_val, y_val, batch_size=5) #len(X_val))

    # Create model, optimizer, loss function, and device
    model = create_model()
    
    optimizer = optim.SGD(model.parameters(), lr=0.000001)
    loss_fn = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # Train model
    train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=50)

    plt.plot(range(len(loss_history)), loss_history)
    plt.show()

    # Run inference
    example_times_of_day = []
    example_temperatures = []
    example_outputs = []
    for i in range(100):
        time_of_day = random.randrange(0, 24)
        temperature = random.uniform(15, 40)
        X = torch.Tensor([[time_of_day, temperature]])
        X = X.to(device)

        output = model(X)

        # print(output)

        example_times_of_day.append(time_of_day)
        example_temperatures.append(temperature)
        example_outputs.append([output.detach().numpy()])

    #plt.scatter(data['Time'], data['Temperature'], data['Demand'])
    #plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    print(data['Demand'])
    ax.scatter(data['Time'], data['Temperature'], data['Demand'], c="#0000ff44")
    ax.scatter(example_times_of_day, example_temperatures, example_outputs, c="#ff0000")
    plt.show()