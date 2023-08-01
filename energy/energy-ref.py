import torch
import torch.nn as nn
import pandas as pd

def train():
    # Load data from CSV file
    data = pd.read_csv('./data/demand_temperature.csv')
    
    # Convert time to time of day
    data['Time'] = pd.to_datetime(data['Time'])
    data['Time'] = data['Time'].dt.hour
    
    # Split data into inputs and targets
    inputs = data[['Time', 'Temperature']].values
    targets = data[['Demand']].values

    # Convert data to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # Use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    print(inputs)
    print(targets)

    # Define model
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Train model
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))

if __name__ == '__main__':
    train()
