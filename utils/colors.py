import numpy as np


def color_shift(im):
    color = list(np.random.random(size=3))
    im[0, :, :][im[0, :, :] >= 0.8] = color[0]
    im[1, :, :][im[1, :, :] >= 0.8] = color[1]
    im[2, :, :][im[2, :, :] >= 0.8] = color[2]
    return im


def color_shift_from_targets(im, targets):
    targets = np.array(targets)
    idx = np.random.choice(range(len(targets)))
    color = [(x + np.random.normal(0, 2))/255 for x in targets[idx]]
    im[0, :, :][im[0, :, :] >= 0.8] = color[0]
    im[1, :, :][im[1, :, :] >= 0.8] = color[1]
    im[2, :, :][im[2, :, :] >= 0.8] = color[2]
    return im
