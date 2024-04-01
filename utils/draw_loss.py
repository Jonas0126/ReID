import numpy as np
import matplotlib.pyplot as plt

def draw_loss(loss, save_path):
    x = np.linspace(1, len(loss), num=len(loss))
    plt.clf()
    plt.plot(x, loss)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.savefig(f'{save_path}')


def draw_acc(acc, save_path):

    x = np.linspace(1, len(acc), num=len(acc))
    plt.clf()
    plt.plot(x, acc)
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.title('acc curve')
    plt.savefig(f'{save_path}')
