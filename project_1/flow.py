from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


def separate_digits(image):
    g = nx.Graph()

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            if image[i, j] == 0:
                continue
            if j < 0.25 * image.shape[1]:
                g.add_edge((-1, -1), (i, j))
            if j > 0.75 * image.shape[1]:
                g.add_edge((-2, -2), (i, j))

            for x, y in [(i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j)]:
                if (
                    x >= 0
                    and y >= 0
                    and x < image.shape[0]
                    and y < image.shape[1]
                    and image[x, y] > 0
                ):
                    g.add_edge((i, j), (x, y), capacity=image[i, j] + image[x, y])

    cut_value, partition = nx.minimum_cut(g, (-1, -1), (-2, -2))

    image1 = np.zeros(shape=image.shape)
    image2 = np.zeros(shape=image.shape)
    for x, y in partition[0]:
        image1[x, y] = image[x, y]
    for x, y in partition[1]:
        image2[x, y] = image[x, y]

    return [image1, image2]


def fix_size(image):
    if image.shape[0] > 28:
        image = image[:28, :]
    if image.shape[1] > 28:
        image = image[:, :28]
    return image
