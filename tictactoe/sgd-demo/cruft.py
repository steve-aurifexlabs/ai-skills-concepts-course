import random
import time
import json

import torch
from torch import nn

log = []
start_time = time.time()
def now():
    return repr((time.time() - start_time) * 1000) + 'ms'

class RecordingLinear(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, id):
        super().__init__()
        self.linear = nn.Linear(input_dimensions, output_dimensions)

    def forward(self, x):
        y = self.linear(x)

        log.append({
            'timestamp': now(),
            'type': 'forward_linear',
            'id': id,
            'x': x,
            'y': y,
        })
        
        return y


def train():
    global log
    
    # Select the best local hardware (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    # Define the model layers
    model = nn.Sequential(
        RecordingLinear(27, 10, 'input_layer'),
        nn.ReLU(),
        RecordingLinear(10, 3, 'output_layer')
    )
    model.to(device)

    # Pick the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for sample_index in range(20):
        global i
        i = 0

        # Play a random game of Tic-Tac-Toe
        board, winner, reason = generate_sample()

        # One-hot encode the data and convert to tensors (batches of 1)
        inputs, actual_outputs = prepare_sample(board, winner, reason)

        # Send the data to the GPU
        inputs, actual_outputs = inputs.to(device), actual_outputs.to(device)
        
        # Reset the gradients before we start
        optimizer.zero_grad()

        # Run the forward pass
        predicted_outputs = model(inputs)

        # Calculate the loss
        loss = criterion(predicted_outputs, actual_outputs)
        
        # Backpropagate the gradients
        loss.backward()

        # Adjust the weights (and biases)
        start_parameters = list(model.parameters())
        optimizer.step()
        adjusted_parameters = list(model.parameters())

        # Print the loss
        if sample_index % 1 == 0:
            print(sample_index, ' Loss: ', loss.item())

        log.append({
            'timestamp': now(),
            'type': 'sample_processed',
            'start_parameters': start_parameters,
            'adjusted_parameters': adjusted_parameters,
            'loss': loss,
            'gradients': [x.grad for x in model.parameters()],
        })

def prepare_sample(board, winner, reason):
    input = []
    output = []

    one_hot_encoding_order = ['x', '-', 'o']

    # One-hot encode the board state as the input
    for piece in board:
        for one_hot_piece in one_hot_encoding_order:
            if one_hot_piece == piece:
                score = 1.0
            else:
                score = 0.0
                
            input.append(score)

    # One-hot encode the winner as the output
    for one_hot_piece in one_hot_encoding_order:
        if one_hot_piece == winner:
            score = 1.0
        else:
            score = 0.0

        output.append(score)
    
    '''
    for reason_index in range(8):
        if reason_index == reason:
            score = 1.0
        else:
            score = 0.0

        output.append(score)
    '''
        
    # Convert to tensors (batch of 1)
    inputs = torch.Tensor([input])
    outputs = torch.Tensor([output])

    return inputs, outputs

def generate_sample():
    # Initialize the game state
    board = [
        '-', '-', '-',
        '-', '-', '-',
        '-', '-', '-',
    ]
    piece = 'x'

    # Keep making moves until the game is over
    while True:

        # Try to place a piece
        while True:
            x = random.randrange(3)
            y = random.randrange(3)
            if board[3 * y + x] == '-':
                board[3 * y + x] = piece
                break

        # Define masks for 3 in a row, col, or diagonal
        win_masks = [
            'aaa??????',
            '???aaa???',
            '??????aaa', 

            'a??a??a??',
            '?a??a??a?',
            '??a??a??a',

            'a???a???a',
            '??a?a?a??',
        ]

        # Check for winner
        for mask_index, mask in enumerate(win_masks):
            win = True
            for square, mask_piece in enumerate(mask):
                if mask_piece == 'a' and board[square] != piece:
                    win = False

            if win:
                return board, piece, mask_index

        # Check if tie
        tie = True
        for x in range(3):
            for y in range(3):
                if board[3 * y + x] == '-':
                    tie = False
        if tie:
            return board, '-', 8  # reason 0-7 are for the mask indices

        # Alternate players
        if piece == 'x':
            piece = 'o'
        else:
            piece = 'x'

if __name__ == '__main__':
    train()
    print(log)