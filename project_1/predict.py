import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from model import model


if __name__ == "__main__":
    model.load_weights("weights/mnist.ckpt")

    for i in range(1):
        x = np.load(f"data/data{i}.npy")
        y = np.load(f"data/lab{i}.npy")
        x = x[:2]
        y = y[:2]

        x = x.reshape(x.shape[0], x.shape[1], 6, x.shape[2] // 6)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3], 1)
        x = tf.image.resize(x, (28, 28))

        z = model.predict(x)
        z = np.argmax(z, axis=1)
        z = z.reshape(-1, 6)
        print(z)
        z = z.sum(axis=1)
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(x[0 + i])
        plt.show()
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            plt.imshow(x[6 + i])
        plt.show()
        print(x.shape, y.shape)
        print(y)
        print(z)
        print(np.mean(y == z))
