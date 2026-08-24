from .mappings import MAPPINGS
from .config import RedshiftConfig
from .constants import SPEED_OF_LIGHT, WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_RANGE_NM
from ..visualization.frames import colormapper, create_frame
from ..physics.redshift_model import calc_inverse_frequency

import numpy as np
import matplotlib.pyplot as plt


def display_map_functions(function = "all", individual = False, flip = False, figsize=(6,6)):
        #Plots examples of all saved mapping functions
        #Plots all functions by default, can be specified to plot 1
        #Plots all maps on one plot by default
        #Set flip to true to invert mapping

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
        #Works but is really slow for some reason? Slightly concerning for actual code speed
        #Plots a fully populated frame
        #Accuracy can be 'quick', 'normal', or 'high'
        #Setting dr value takes priority over accuracy
       
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