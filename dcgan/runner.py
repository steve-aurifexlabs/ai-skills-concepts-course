import subprocess
import os

experiments = ['squares_adam', 'squares_sgd', 'squares_lion']

while True:
    for experiment in experiments:
        experiment_path = os.path.join('experiments', experiment)
        subprocess.run(['python3', 'train.py'], cwd=experiment_path)