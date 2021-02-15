import numpy as np


def cartesian_product(s1, s2):
    return np.transpose([np.tile(s1, len(s2)),
                         np.repeat(s2, len(s1))])
