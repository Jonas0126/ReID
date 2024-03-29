import numpy as np
import matplotlib.pyplot as plt

def draw_loss(loss, save_path):
    x = np.linspace(1, len(loss), num=len(loss))
    plt.plot(x, loss)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.savefig(f'{save_path}')

