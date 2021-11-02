import numpy as np
from matplotlib import pyplot as plt
import os

'''<==plot loss curves==>'''
def plot_curve(train_loss, val_loss):
    plt.figure(figsize=(20, 8))
    x = np.arange(1, len(train_loss)+1)
    plt.plot(x, train_loss, color="r", linestyle="-", linewidth=1.0, marker='o', markersize=2, label='training')
    plt.plot(x, val_loss, color='g', linestyle='-', linewidth=1.0, marker='o', markersize=2, label='validation')
    plt.xlabel("Epochs")
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join('results/loss.jpg'))
