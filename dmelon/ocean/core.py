import numpy as np


class DispersionRelation:
    """
    Dispersion Relation
    """

    @staticmethod
    def low_freq(m):
        BETA = 2.29e-11
        C = 2.7
        CONST1 = (BETA * C) ** (1 / 2)
        CONST2 = (BETA / C) ** (1 / 2)
        k = np.arange(-10, 10, 0.1)
        w = -k / (2 * m + 1 + k ** 2)
        return w * CONST1 * 60 * 60 * 24 / (2 * np.pi), k * CONST2 * 110e3 / (2 * np.pi)
