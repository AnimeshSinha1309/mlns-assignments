import numpy as np
from matplotlib import pyplot as plt
import tqdm.auto as tqdm

from flow import separate_digits, fix_size


x = np.load("data/data0.npy")
y = np.load("data/lab0.npy")


def segment(image):
    threshold = 1
    visited = np.zeros(image.shape, dtype=np.int32)

    def dfs(i, j, val):
        if (
            i < 0
            or i >= image.shape[0]
            or j < 0
            or j >= image.shape[1]
            or image[i][j] < threshold
            or visited[i, j] != 0
        ):
            return
        visited[i, j] = val
        dfs(i + 1, j, val)
        dfs(i - 1, j, val)
        dfs(i, j + 1, val)
        dfs(i, j - 1, val)

    current_component = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if visited[i][j] == 0 and image[i][j] >= threshold:
                current_component += 1
            dfs(i, j, current_component)

    lower_x = np.full(shape=current_component, fill_value=1000)
    lower_y = np.full(shape=current_component, fill_value=1000)
    upper_x = np.full(shape=current_component, fill_value=-1000)
    upper_y = np.full(shape=current_component, fill_value=-1000)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if visited[i, j] != 0:
                lower_x[visited[i, j] - 1] = min(lower_x[visited[i, j] - 1], i)
                lower_y[visited[i, j] - 1] = min(lower_y[visited[i, j] - 1], j)
                upper_x[visited[i, j] - 1] = max(upper_x[visited[i, j] - 1], i)
                upper_y[visited[i, j] - 1] = max(upper_y[visited[i, j] - 1], j)

    digits = []
    for i in range(current_component):
        digit = image[lower_x[i] : upper_x[i], lower_y[i] : upper_y[i]]
        if len(lower_x) == 3 and (digit.shape[0] > 28 or digit.shape[1] > 28):
            digits.extend(separate_digits(digit))
        else:
            digits.append(digit)

    digits = list(filter(lambda x: np.sum(x > 0) > 10, digits))

    images = []
    for digit in digits:
        if digit.shape[0] > 28 or digit.shape[1] > 28:
            digit = fix_size(digit)
        pad_shape = np.array([28, 28]) - np.array(digit.shape)
        digit = np.pad(
            digit,
            (
                (pad_shape[0] // 2, (pad_shape[0] + 1) // 2),
                (pad_shape[1] // 2, (pad_shape[1] + 1) // 2),
            ),
        )
        images.append(digit)

    return np.stack(images, axis=0)


if __name__ == "__main__":
    count = 0
    for i in tqdm.trange(len(x)):
        res = segment(x[i])
        if res != 4:
            count += 1

    print(count, count * 100 / len(x))
