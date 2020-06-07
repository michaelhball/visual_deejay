import math
import matplotlib.pyplot as plt


__all__ = ["angle_between_two_lines", "plot_one", "plot_two"]


def angle_between_two_lines(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """

    x1, y1 = v1
    x2, y2 = v2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def plot_one(img, cmap='viridis'):
    """

    :param img:
    :param cmap:
    :return:
    """

    plt.imshow(img, cmap=cmap)
    plt.show()


def plot_two(img1, img2, cmap='viridis'):
    """

    :param img1:
    :param img2:
    :param cmap:
    :return:
    """

    fig = plt.figure()
    _ = fig.add_subplot(1,2,1).imshow(img1, cmap=cmap)
    _ = fig.add_subplot(1,2,2).imshow(img2, cmap=cmap)
    plt.show()
