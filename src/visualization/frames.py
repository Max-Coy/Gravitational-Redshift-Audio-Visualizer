import numpy as np
import matplotlib
from ..core.constants import WAVELENGTH_MIN_NM, WAVELENGTH_RANGE_NM, WAVELENGTH_MAX_NM, SPEED_OF_LIGHT
from ..audio.fft_pipeline import grab_freq

from ..core.mappings import MAPPINGS
from ..physics.redshift_model import find_energy, map_frequency_to_color, freq_to_angle
from ..core.config import RedshiftConfig

def build_frames(data, N, xf, max_H, config: RedshiftConfig):
    """
    Converts windowed audio signal into frame representations.

    Parameters
    ----------
    data : np.ndarray
        Mono audio signal
    N : int
        Samples per frame

    Returns
    -------
    list
        List of frames for rendering
    """
    num_frames = len(data) // N
    
    frames = []

    for i in range(num_frames):
        start = i * N
        end = start + N

        window = data[start:end]

        freqs, heights = grab_freq(window, N, xf, config)
        frame = create_frame(freqs, heights, max_H, config)

        frames.append(frame)

    return frames


def create_frame(
    freq: np.ndarray,
    h: np.ndarray,
    max_H,
    config: RedshiftConfig,
):
    """
    Construct a single animation frame from frequency + amplitude data.
    """

    angles = freq_to_angle(freq, config) # Map frequency to angular position on circle.

    # Radial Displacement
    offset_mapper = MAPPINGS[config.volume_offset_mapping]
    # config.volume_offset is a bool and controls whether the later half is used
    r = config.r - config.volume_offset * (config.volume_offset_max * offset_mapper(h / max_H))

    # Alpha channel
    alpha_mapper = MAPPINGS[config.volume_alpha_mapping]
    ar = config.volume_alpha_range[1] - config.volume_alpha_range[0]
    # config.volume_alpha is also a bool doing the same thing
    alpha = config.volume_alpha_range[1] - config.volume_alpha * (ar * MAPPINGS["flip"](alpha_mapper, h / max_H))

    # Plot Geometry
    
    X = r * np.cos(angles)
    Y = r * np.sin(angles)
    # adding fake second points so that we can draw lines
    dX = config.dr * np.cos(angles) 
    dY = config.dr * np.sin(angles)

    # Frequency adjustments (redshift + mapping)
    # config.frequency_flip is another bool for more branchless logic
    f = (config.max_frequency - freq) * config.frequency_flip + freq * (not config.frequency_flip)
    f = f % (config.max_frequency / config.colormap_periods)
    # can you tell that I had just watched a video on branchless programming when I was writing all of this...
    f = f * (not config.colormap_mirror) + np.abs(f - config.max_frequency/(config.colormap_periods*2)) * config.colormap_mirror #checking for mirror

    omega = SPEED_OF_LIGHT / map_frequency_to_color(f, WAVELENGTH_MIN_NM + config.colormap_range[0] * WAVELENGTH_RANGE_NM, WAVELENGTH_MAX_NM * config.colormap_range[1], config)

    E = find_energy(omega, config)

    return X, dX, Y, dY, E, alpha


def wave_position(wl: float):
    """
    Normalize wavelength into [0, 1] for colormap sampling.
    """

    return np.clip((wl - WAVELENGTH_MIN_NM) / WAVELENGTH_RANGE_NM, 0, 1)


def colormapper(lam: float, config, w2r_table):
    """
    Convert wavelength to RGB color.
    """

    if config.colormap == "rainbow":
        idx = np.abs(w2r_table[:, 0] - lam).argmin()
        return w2r_table[idx][1:]

    cmap = matplotlib.cm.get_cmap(config.colormap)

    return cmap(wave_position(lam))
