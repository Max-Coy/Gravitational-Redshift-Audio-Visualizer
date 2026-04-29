
import numpy as np
import matplotlib.pyplot as plt

def anim_plotter(inputs):
    fn, plots = inputs    
    fig = plt.figure(figsize=(6,6))
    for i in range(len(fn)):
        for arr in plots[i]:#Plotting each line in frame
                plt.plot(arr[0][0],arr[0][1],color=arr[-2],alpha=arr[-1])
        plt.xlim(-6,6) #setting up axes and jawn
        plt.ylim(-6,6)
        plt.scatter([0],[0],color='black',linewidth=1)
        plt.xticks([])
        plt.yticks([]);
        fig.savefig(fn[i])
        plt.cla();
    return 1
