import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm.auto as tqdm

from model import model
from segment import segment


if __name__ == "__main__":
    model.load_weights("weights/mnist.ckpt")

    for i in range(3):
        x = np.load(f"data/data{i}.npy")
        y = np.load(f"data/lab{i}.npy")

        iterator = tqdm.trange(len(x))
        total, correct = 0, 0
        
        for i in iterator:
            z = model.predict(segment(x[i]))
            z = np.argmax(z, axis=1)
            z = np.sum(z)

            if y[i] == z:
                correct += 1
            total += 1
            iterator.set_postfix(accuracy=correct / total)
