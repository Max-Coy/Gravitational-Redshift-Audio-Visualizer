import numpy as np
from ..core.mappings import MAPPINGS
from ..core.config import RedshiftConfig

def freq_to_angle(f: float, config: RedshiftConfig) -> float:
    """
    Map frequency to angular position on circle.
    """

    mapper = MAPPINGS[config.frequency_mapping]

    return 2 * np.pi * mapper(f * (1 / config.max_frequency))

def find_energy(w: float, config: RedshiftConfig) -> float:
    """
    Compute photon energy under gravitational potential.
    """

    return w * np.sqrt(1 - 2 * config.M / config.r)

def calc_inverse_frequency(e: float, r: float, config: RedshiftConfig) -> float:
    """
    Apply gravitational redshift correction.
    """

    return np.sqrt(1 - 2 * config.M / r) / e

def map_frequency_to_color(
    freq: float,
    start: float,
    stop: float,
    config: RedshiftConfig,
) -> float:
    """
    Map frequency into wavelength/color space.
    """

    pos = freq * (1 / config.max_frequency) * config.colormap_periods

    mapper = MAPPINGS[config.frequency_mapping]

    mf = mapper(pos)

    return start + (stop - start) * mf