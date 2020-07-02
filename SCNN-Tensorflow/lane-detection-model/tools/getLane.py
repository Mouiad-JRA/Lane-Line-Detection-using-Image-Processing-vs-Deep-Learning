"""
Created on Sat May 30 22:30:44 2020

@author: Mouiad
"""
import numpy as np


def getLane(score):
    # Calculate lane position from a probmap.
    thr = 0.3
    coordinate = np.zeros((1, 18))
    for i in range(18):
        lineId = np.uint16(np.round(288.0 - i * 20.0 / 590.0 * 288.0))
        line = score[lineId - 1, :]
        value, index = np.amax(line), np.where(line == np.amax(line))[0][0]
        if value / 255.0 > thr:
            coordinate[0, i] = index + 1  # TODO be sure about the '+1'
    if np.sum(coordinate > 0) < 2:
        coordinate = np.zeros((1, 18))
    return coordinate
