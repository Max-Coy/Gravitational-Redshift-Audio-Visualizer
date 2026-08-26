from .mappings import MAPPINGS
from .config import RedshiftConfig
from .constants import SPEED_OF_LIGHT, WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_RANGE_NM
from ..visualization.frames import colormapper, create_frame
from ..physics.redshift_model import calc_inverse_frequency, map_frequency_to_color, find_energy

import numpy as np
import matplotlib.pyplot as plt

def display_progress_bar(current_progress: float, character_length: int = 10):
    """
    Displays a simple progress bar
    """
    empty_square = '░'
    half_square = '▒'
    full_square = '█'
    if current_progress < 0:
        print(f"Error, current progress is {current_progress}\nPlease give a number larger than 0")
        return
    elif current_progress >= 1:
        print(full_square * character_length + '\r', end = '')
        return
        
    total_states = character_length * 2 + 1
    steps = 1 / total_states
    current_state = current_progress // steps

    if current_state == total_states: #Full progress bar is reserved for 100% completion
        current_state -= 1

    filled_squares = int(current_state - current_state // 2)
    end_in_full = ((current_state - 1) % 2) == 1
    if end_in_full:
        progress = full_square * filled_squares
    else:
        progress = full_square * (filled_squares - 1) + half_square

    progress += empty_square * (character_length - filled_squares)

    print(progress + '\r', end = '')


def display_map_functions(function = "all", individual = False, flip = False, figsize=(6,6)):
        """
        Plots examples of all saved mapping functions
        Plots all functions by default, can be specified to plot 1
        Plots all maps on one plot by default
        Set flip to true to invert mapping
        """
        xp = np.linspace(0,1,500)
        if function == "all": #plotting all maps
            funcs = list(MAPPINGS.keys())
        else: #plotting one map
            funcs = [function]

        fig = plt.figure(figsize=figsize)
        for f in funcs:
            if f != "flip":
                if flip:
                    y = MAPPINGS['flip'](MAPPINGS.func_dict[f],xp)
                else:
                    y = MAPPINGS[f](xp)
                plt.plot(xp,y,label=f)
                if individual:
                    plt.legend()
                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    plt.show()
                    fig = plt.figure(figsize=figsize)

        if not individual:
            plt.legend()
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.show()
        return

def display_full_frame(config: RedshiftConfig, window = "full", accuracy="normal", dr = None, alpha = 1, save = False):
        """
        Plots a fully populated frame
        Accuracy can be 'quick', 'normal', or 'high'
        Setting dr value takes priority over accuracy
        Saving is currently disabled
       """
        
        w2r = np.genfromtxt("src/w2r_blend.csv", delimiter=" ") #not clean, might want to refactor colormapper function?

        if dr == None:
            if accuracy == 'high':
                dr = config.dr / 2
            elif accuracy == 'normal':
                dr = config.dr
            else:
                dr = config.dr * 2

        rdr = dr/config.r
        maxFI = 1/config.max_frequency
        l = config.min_frequency - (config.min_frequency % config.angular_resolution)
        h = config.max_frequency - (config.max_frequency % config.angular_resolution)

        full_freq = np.arange(l,h+1,config.angular_resolution)


        if window == "reduced":
          sixth = full_freq.shape[0] // 16 #reducing full freq window to be -pi/8 : pi/8
          full_freq = np.hstack([full_freq[:sixth],full_freq[-sixth:]])

        heights = np.ones(len(full_freq)) #* maxH

        #Getting all possible active frequencies
        Full_Frame = [create_frame(full_freq, heights, 1, config)]
        while not ((np.abs(Full_Frame[0][0]) > config.xlim) + (np.abs(Full_Frame[0][2]) > config.ylim)).all():
            for i in range(len(Full_Frame)):
                Full_Frame[i][0] += Full_Frame[i][1]
                Full_Frame[i][2] += Full_Frame[i][3]
            Full_Frame.append(create_frame(full_freq,heights, 1, config))

        #Plotting frequencies
        fig = plt.figure(figsize = config.figsize)
        for i in range(len(Full_Frame)):#
            for j in range(len(Full_Frame[i][0])):#
                xs = [Full_Frame[i][0][j], Full_Frame[i][0][j]+ Full_Frame[i][1][j] ]
                ys = [Full_Frame[i][2][j], Full_Frame[i][2][j]+ Full_Frame[i][3][j] ]

                dist = 1 + rdr * (len(Full_Frame) - i)
                wnI = calc_inverse_frequency(Full_Frame[i][4][j], dist, config)
                lam = wnI * SPEED_OF_LIGHT
                plt.plot(xs, ys, color = colormapper(lam, config, w2r), alpha=alpha)

        plt.xlim(-config.xlim, config.xlim)
        plt.ylim(-config.ylim, config.ylim)
        plt.scatter([0],[0],color='black',linewidth=1)
        plt.axis('off');
        # if save:
        #     plt.savefig(a.filepath.split('.')[0] + "_Full_Frame.png", format='png',bbox_inches='tight');
        plt.show();
        return

def display_colormap(config: RedshiftConfig, save = False):
        """
        A faster alternative to display_full_frame()
        Makes a simple plot of the minimum and maximum redshift for a given wavelength based on the 
        current config. Saving is currently disabled
        """

        w2r = np.genfromtxt("src/w2r_blend.csv", delimiter=" ") #same issue as display full frame

        x_points = np.linspace(330,890,1236)
        for x in x_points: #plotting background
            plt.hlines(x, 0, 1, color = colormapper(x, config, w2r))

        rI = config.r - config.volume_offset_max
        l = config.min_frequency - (config.min_frequency % 20)
        h = config.max_frequency - (config.max_frequency % 20)
        full_freq = np.arange(l, h+1, 20)
        f = (config.max_frequency - full_freq) * config.frequency_flip + full_freq * (not config.frequency_flip) #checking for freq. flip
        f = f % (config.max_frequency / config.colormap_periods) #checking for periods
        f = f * (not config.colormap_mirror) + np.abs(f - config.max_frequency/(config.colormap_periods*2)) * config.colormap_mirror #checking for mirror

        omega =  SPEED_OF_LIGHT / map_frequency_to_color(f, 
                                                         WAVELENGTH_MIN_NM + WAVELENGTH_RANGE_NM * config.colormap_range[0],
                                                         WAVELENGTH_MAX_NM * config.colormap_range[1],
                                                         config)
        E = find_energy(omega, config)
        dist = rI
        wnI = calc_inverse_frequency(E, dist, config) 

        lam = wnI * SPEED_OF_LIGHT
        lam[lam < x_points[0]] = x_points[0] * 0.995
        lam[lam > x_points[-1]] = x_points[-1] * 1.005
        xp = (full_freq / config.max_frequency) % 1

        plt.plot(xp[xp.argsort()], lam[xp.argsort()], color='black', label= 'Minimum Redshift')

        ang = xp * 2 * np.pi
        a1 = np.cos(ang)
        a2 = np.sin(ang)
        md = 6 / (np.vstack(np.abs([a1, a2])).T).max(axis=1) #max distance

        for i in range(len(E)):
            wnI[i] = calc_inverse_frequency(E[i], md[i], config) 

        lam = wnI * SPEED_OF_LIGHT
        lam[lam < x_points[0]] = x_points[0] * 0.995
        lam[lam > x_points[-1]] = x_points[-1] * 1.005
        xp = (full_freq / config.max_frequency) % 1

        plt.plot(xp[xp.argsort()], lam[xp.argsort()], linestyle='dashed', color='black', label='Maximum Redshift')
        plt.xlim(0,1)
        plt.ylim(x_points[0], x_points[-1])
        plt.xticks([]);
        plt.text(0.0, x_points[0] - 0.15 * x_points[0], "Low Frequency")
        plt.text(0.77, x_points[0] - 0.15 * x_points[0], "High Frequency")
        plt.ylabel('Output Color')
        plt.legend()
        plt.yticks([]);
        plt.title('Output Color Mapping');
        # if save:
        #     plt.savefig(a.filepath.split('.')[0] + "_Colormap.png", format='png',bbox_inches='tight');
        plt.show();
        return