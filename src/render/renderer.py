import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from multiprocessing import Pool
from ..visualization.frames import create_frame
from ..physics.redshift_model import calc_inverse_frequency
from ..core.constants import SPEED_OF_LIGHT
from ..visualization.frames import colormapper
from ..core.config import RedshiftConfig
from ..core.mappings import MAPPINGS

def build_plot_frame(
    i,
    Frames,
    frameWindow,
    rdr,
    config,
    w2r
):
    """
    Expands a single time-step index into drawable line segments.
    - Removes frames that are outside of viewing window
    - Computes photon propagation
    - Applies redshift + wavelength mapping
    - Produces drawable segments for matplotlib

    Returns
    -------
    plotF : list
        Each element = [segment, color, alpha]
    """

    lastFrame = max(i - frameWindow, 0)
    plotF = []

    for j in range(i, lastFrame - 1, -1):

        frame = Frames[j]

        xs_base = frame[0]
        xs_step = frame[1]
        ys_base = frame[2]
        ys_step = frame[3]
        energies = frame[4]
        alphas = frame[5]

        time_offset = i - j

        for k in range(len(xs_base)):

            xs = [
                xs_base[k] + xs_step[k] * time_offset,
                xs_base[k] + xs_step[k] * (time_offset + 1),
            ]

            ys = [
                ys_base[k] + ys_step[k] * time_offset,
                ys_base[k] + ys_step[k] * (time_offset + 1),
            ]

            # Skip out-of-bounds early
            if abs(xs[0]) >= config.xlim or abs(ys[0]) >= config.ylim:
                continue

            dist = 1 + rdr * (time_offset + lastFrame)

            wnI = calc_inverse_frequency(energies[k], dist, config)
            lam = wnI * SPEED_OF_LIGHT

            color = colormapper(lam, config, w2r)

            plotF.append([
                np.array([xs, ys]),
                color,
                alphas[k],
            ])

    # fallback (prevents empty frame crash in renderer)
    if not plotF:
        plotF.append([
            np.array([[0, 0], [0, 0]]),
            "black",
            1,
        ])

    return plotF


def build_render_context(config, analysis):
    """
    Converts audio analysis + config into rendering parameters.
    """

    # Normalize frequency bounds to config constraints
    min_freq = max(config.min_frequency, analysis[0])
    max_freq = min(config.max_frequency, analysis[1])

    # Snap to angular resolution (important for visual stability)
    max_freq -= max_freq % config.angular_resolution
    min_freq -= min_freq % config.angular_resolution

    # Frame propagation scaling (geometric time evolution)
    redshift_rate = config.dr / config.r

    # Window size for time overlap of frames
    config.max_frequency = max_freq
    frame_window = find_linger(config, min_freq, max_freq, analysis[2])

    return (
        min_freq,
        max_freq,
        redshift_rate,
        frame_window,
    )


def find_linger(config, min_freq, max_freq, max_H):
    """
    Determine the max number of timesteps a frame remains visible
    before leaving viewport bounds.
    """

    full_freq = np.arange(min_freq, max_freq + 1, config.angular_resolution)

    heights = np.ones(len(full_freq)) * max_H

    X, dX, Y, dY, _, _ = create_frame(full_freq, heights, max_H, config)

    index = 1
    # Running until every beam of light has an X or Y coordinate larger than viewport limits
    while not ( (np.abs(X) > config.xlim) + (np.abs(Y) > config.ylim) ).all():

        X += dX
        Y += dY

        index += 1

    return index

def pool_plotter(cores: int, filenames: list[str], plots: list, config: RedshiftConfig):
    """
    Render frames in parallel batches using multiprocessing.

    Each worker process handles a subset of frames and writes them to disk.
    """

    if cores <= 1:
        # Fallback: single-threaded execution (avoids multiprocessing overhead)
        for fn, plot_group in zip(filenames, plots):
            anim_plotter(([fn], [plot_group], config))
        return None

    pool = Pool(processes = cores)

   
    chunk_size = len(filenames) // cores
    if chunk_size == 0:
        chunk_size = 1

    data = []


    for i in range(0, len(filenames), chunk_size):
        data.append(
            (
                filenames[i:i + chunk_size],
                plots[i:i + chunk_size],
                config
            )
        )


    results = pool.map(anim_plotter, data)


    pool.close()
    pool.join()

    return results



def anim_plotter(inputs):
    """
    Takes a batch of filenames + precomputed plot data,
    renders each frame, and writes it to disk.

    Inputs should take the form of: 
        [list of file names
         list of all plot data
         config dataclass]
    """

    fn_list, plots, config = inputs

    for i in range(len(fn_list)):

        fig, ax = plt.subplots(figsize=config.figsize)

        frame = plots[i]

        for arr in frame:
            ax.plot(
                arr[0][0],
                arr[0][1],
                color=arr[-2],
                alpha=arr[-1],
            )

        ax.set_xlim(-config.xlim, config.xlim)
        ax.set_ylim(-config.ylim, config.ylim)

        ax.scatter([0], [0], color="black", linewidth=1)

        ax.set_xticks([])
        ax.set_yticks([])

        
        fig.savefig(fn_list[i], bbox_inches="tight")

        plt.close(fig)

    return 1

def cleanup_render_directory(temp_dir: Path, remove_directory: bool = True):
    if not temp_dir.exists():
        return
    
    if remove_directory:
        shutil.rmtree(temp_dir)