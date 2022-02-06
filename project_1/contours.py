import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as pg

import cv2 as cv
from flow import fix_size, separate_digits


def segment(grayscale_image):
    original_image = grayscale_image.copy()
    _threshold_value, binary_image = cv.threshold(grayscale_image, 0, 255, cv.THRESH_OTSU)
    contours, _hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    digits, areas = [], []

    for _, c in enumerate(contours):
        bounding_rectangle = cv.boundingRect(c)
        rect_x = bounding_rectangle[0]
        rect_y = bounding_rectangle[1]
        rect_w = bounding_rectangle[2]
        rect_h = bounding_rectangle[3]

        rect_area = rect_w * rect_h
        if rect_area > 10:
            color = (0, 255, 0)
            cv.rectangle(original_image, (int(rect_x), int(rect_y)),
                        (int(rect_x + rect_w), int(rect_y + rect_h)), color, 2)
            current_crop = binary_image[rect_y:rect_y+rect_h,rect_x:rect_x+rect_w]
            digits.append(current_crop)
            areas.append(rect_w * rect_h)

    if len(digits) > 4:
        digits = [digits[idx] for idx in np.argsort(areas)[-4:]]
    elif len(digits) < 4:
        target_digit = np.argmax(areas)
        results = separate_digits(digits[target_digit])
        digits[target_digit] = results[0]
        digits.extend(results[1:]) 

    images = []
    for digit in digits:
        if digit.shape[0] > 28 or digit.shape[1] > 28:
            digit = fix_size(digit)
        pad_shape = np.array([28, 28]) - np.array(digit.shape)
        digit = np.pad(digit, ((pad_shape[0] // 2, (pad_shape[0] + 1) // 2), (pad_shape[1] // 2, (pad_shape[1] + 1) // 2)))
        images.append(digit)

    images = np.stack(images, axis=0)
    return images


if __name__ == "__main__":
    data = np.load("data/data0.npy")
    res = segment(data[0])
    print(res.shape)
    for i in range(len(res)):
        plt.subplot(1, len(res), i + 1)
        plt.imshow(res[i])
    plt.show()
