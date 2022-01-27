import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tqdm.auto as tqdm

from model import model
from segment import segment


if __name__ == "__main__":
    model.load_weights("weights/mnist.ckpt")

    for i in range(1):
        x = np.load(f"data/data{i}.npy")
        y = np.load(f"data/lab{i}.npy")

        iterator = tqdm.trange(len(x))
        total, correct = 0, 0
        
        for i in iterator:
            image = x[i].astype(np.int32)

            segmented_images = segment(image)
            p = model.predict(segmented_images)
            p = np.argmax(p, axis=1)
            z = np.sum(p)

            if y[i] == z:
                correct += 1
            # else:
            #     for i in range(len(segmented_images)):
            #         plt.subplot(1, len(segmented_images), 1 + i)
            #         plt.imshow(segmented_images[i])
            #         plt.title(f"{p[i]}")
            #         plt.axis('off')
            #     plt.show()

            total += 1
            iterator.set_postfix(accuracy=correct / total)
