import numpy
from typing import Tuple

def generate_whiteNoise(sizeIMG: Tuple[int, int, int], noiseLVL: float):
    noiseOut = numpy.random.normal(0, noiseLVL, size=sizeIMG)

    return noiseOut
