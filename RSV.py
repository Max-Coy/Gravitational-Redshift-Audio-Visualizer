#RedShift Visualizer V0.1
#Max Coy, 2/26
#Python Code for converting audio into a visualized .mp4 format using the redshift caused
#by gravity
#Beefy boy, slow boy
#愛しているぞ

#Importing Libraries
import numpy as np
import datetime
import ffmpeg
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
from poolPlot import anim_plotter #custom python file
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import blackman
import os
from pydub import AudioSegment

#############################################################################################################
#Colormapping functions
#Linear: pos
def linear(x):
    return x

#sqrt: np.sqrt(pos)
def square_root(x):
    return np.sqrt(x)

#4th root: np.pow(pos, 0.25)
def fourth_root(x):
    return np.power(x,0.25)

#exp: np.exp(pos - n) - np.exp(-n); n = np.log(np.e - 1)
def exponential(x):
    n = np.log(np.e - 1)
    return np.exp(x - n) - np.exp(-n)

#10^: np.pow(10, pos - n) - np.pow(10,-n); n = np.log10(10 - 1)
def exponential_ten(x):
    n = np.log10(10 - 1)
    return np.power(10, x - n) - np.power(10,-n)

#sin: np.sin(np.pi * pos * 0.5)
def sine(x):
    return np.sin(np.pi * x * 0.5)

#tan: 0.09 * np.tan(2.60*pos + 1.82) + 0.52;this one is pretty funky ngl
def tangent(x):
    return 0.09 * np.tan(2.60*x + 1.82) + 0.52

#sigmoid: 1 / (1 + np.exp(-8 * (pos - 0.5))); not full range / is adjustable
def sigmoid(x):
    return 1 / (1 + np.exp(-8 * (x - 0.5)))

#jet sigmoid: 1 / (1 + np.exp(-6 * (pos - 0.5)))
def sigmoid2(x):
    return 1 / (1 + np.exp(-6 * (x - 0.5)))

#flip: 1 - (any mapping)
def flip(f,x):
    return 1 - f(x)

#############################################################################################################
#For multicore plotting
def poolPlotter(cores, filenames, plots):
    pool = Pool(processes = cores)
    chunk = int(len(filenames)/cores)#assigning 1 core for each process
    extra = int(len(filenames) - cores*chunk) #to account for any rounding error
    data = []
    if chunk != 0:
        for i in range(np.floor(len(filenames) / chunk).astype(int)):
            data.append([filenames[chunk*i:chunk*(i+1)], plots[chunk*i:chunk*(i+1)]])
        if extra:
            data[-1] = [filenames[chunk*i:], plots[chunk*i:]]
        results = pool.map(anim_plotter, data)
    else:
        for i in range(len(filenames)):
            data = [filenames[i],plots[i]]
            anim_plotter(data)
    del pool

    return results#, end-start, iterations

#############################################################################################################
#Main classwork
class redshift_animation():

    def __init__(self, inputpath, outputpath = "", framerate = 24, cores = 1, resolution = 0.2, angular_resolution = 40, figsize = (6,6)):

        #Setting input and outpath filepaths
        self.filepath = inputpath #should be the filepath to input .wav file
        if self.filepath.split(".")[1] == "mp3":
          self.mp3ToWav()
          
        if outputpath == "" or type(outputpath) != str: #outputpath not specified, using input path as basis
            self.output = inputpath.split('.')[0] + '_redshift_animation.mp4'
        else:
            self.output = outputpath #filepath that will be used when creating final render file
        self.temp_output = self.output[:-4]+ "_temp.mp4" #Temporary output to used to make video before adding audio
        #Setting core parameters
        self.samplerate = framerate #sampling of the input audiofile will determine framerate of output file
        self.cores = cores #controls multiprocessing module / how many cores to use when plotting
        self.batchSize = 60 #how many frames will be drawn at once

        #Presetting extra parameters, will have other functions for modifying these values
        self.colormap = 'jet'#'rainbow' #mapping for wavelengths to color
        self.w2r = np.genfromtxt("w2r_blend.csv", delimiter=' ')
        self.colormap_range = np.array([0, 1]) #specifies which part of the mappings to use
        self.colormap_periods = 1 #setting to higher integer powers causes the colormap to repeat periodically
        self.colormap_mirror = False #Mirror top and bottom halves of color map
        self.frequency_mapping = 'linear' #Controls how audio frequencies are mapped to wavelengths
        self.frequency_flip = False #flips low and high ends of frequency mapping
        self.volume_offset = True #If True light beams are offset radially depending on signal strength
        self.volume_offset_max = 1.5 #Controls the maximum volume offset (should not be larger than r)
        self.volume_offset_mapping = '4th_root' #mapping function for volume offset
        self.volume_alpha = True #if True, weaker signals (i.e. "quieter" frequencies) will have alpha channel value reduced
        self.volume_alpha_range = np.array([0.5,1]) #set minimum and maximum alpha channel values, should be between 0 and 1, used regardless of value for self.volume_alpha
        self.volume_alpha_mapping = "sigmoid2" #mapping function for alpha values

        self.M = 0.15 #Controls rate of redshifting (mass of center object)
        self.c = 3e8 #setting speed of light
        self.r = 2 #starting radius of light beams
        self.dr = resolution #controls how long light beams are drawn (lower # => slower light => more active frames/higher resolution)
        self.angular_resolution = angular_resolution #controls frequency bin size
        self.min_frequency = 0 #modifying before calling render function will set a lower limit on plotted frequencies
        self.max_frequency = 3e5 #modifying before calling render function will set an upper limit on plotted frequencies
        self.max_FI = 1/self.max_frequency
        self.max_H = 1#maximum volume recorded
        self.min_Volume = 1e-10
        self.xf = None #x-values for fft

        self.figsize = figsize #figure size used by matplotlib when plotting
        self.xlim = 6 #plotting bounds of figure in x direction
        self.ylim = 6 #plotting bounds of figure in y direction

        self.temp_dirs = 'temp_img_holder' #Temporary directory which will utilized during code execution
        self.remove_temp_dirs = True #leaving True will cause temporary directory to be deleted after code execution

        self.print_global_progress = True #leaving true will cause code to print whenever it enters a new stage during render
        self.print_local_progress = False #Setting true will cause code to print progress while creating animation frames

        self.func_dict = {"linear" : linear,
             "sqrt" : square_root,
            "4th_root": fourth_root,
            "e^x" : exponential,
            "10^x" : exponential_ten,
            "sin" : sine,
            "tan" : tangent,
            "sigmoid" : sigmoid,
            "sigmoid2" : sigmoid2,
            "flip" : flip} #stores mapping functions

    def mp3ToWav(self): #Converts .mp3 input to .wav input
      print(".mp3 input detected, attempting to convert to .wav")
      try:
          dst = self.filepath.split(".")[0] + ".wav"
          sound = AudioSegment.from_mp3(self.filepath)
          sound.export(dst, format="wav")
          self.filepath = dst
          print("Conversion Sucessful")
          return True
      except Exception as e:
          print(f"An error occurred: {e}")
          return False

    def freq2ang(self, f): #maps initial frequency to output angle
        return 2 * np.pi * self.func_dict[self.frequency_mapping](f * self.max_FI)

    def grabFreq(self, data, window): #grabs all relevant fft data from active frame
        wb = blackman(len(data))
        yf = fft(data * wb)#taking fft
        y =  2.0/window * np.abs(yf[0:window//2]) #converting fft to real values
        y[y<5e-5] = 0 #murdering noise under a threshold
        indii = find_peaks(y)[0]
        arr = np.array(list(set(self.xf[indii] - (self.xf[indii] % self.angular_resolution))))
        height = (2.0/window * np.abs((y[indii])[0:window//2]))#sample starting radius
        #print(len(height), len(height[height > self.min_Volume]), height.min())
        return arr[height > self.min_Volume], height[height > self.min_Volume] #returns all values above minimum threshold

    def findE(self, w): #finds initial energy value
        return w * np.sqrt(1 - 2*self.M/self.r)

    #Calculates adjusted inverse photon frequency due to redshift
    def calcWI(self, e,r):
        return np.sqrt(1 - 2*self.M/r) / e

    def mapFreq(self, freq, start,stop): #maps current frequency to output color
        #Start = min. output lambda, stop = max. output lambda
        pos = (freq * self.max_FI*self.colormap_periods)
        mf =  self.func_dict[self.frequency_mapping](pos)
        return start + (stop - start) * mf

    def createFrameC(self, freq, h):
        angles = (self.freq2ang(freq))
        r = self.r - self.volume_offset * self.volume_offset_max * self.func_dict[self.volume_offset_mapping](h/self.max_H)
        ar = self.volume_alpha_range[1] - self.volume_alpha_range[0] #next line was getting a little long lmao
        alpha = self.volume_alpha_range[1] - self.volume_alpha * ar * self.func_dict['flip'](self.func_dict[self.volume_alpha_mapping],h/self.max_H)
        X = r*np.cos(angles)
        Y = r*np.sin(angles)
        dX = self.dr * np.cos(angles) #adding a fake second point so that we can draw a line
        dY = self.dr * np.sin(angles)

        f = (self.max_frequency - freq) * self.frequency_flip + freq * (not self.frequency_flip) #checking for freq. flip
        f = f % (self.max_frequency / self.colormap_periods) #checking for periods
        f = f * (not self.colormap_mirror) + np.abs(f - self.max_frequency/(self.colormap_periods*2)) * self.colormap_mirror #checking for mirror

        omega =  self.c/self.mapFreq(f,330 + self.colormap_range[0]*560,890*self.colormap_range[1])

        E = self.findE(omega) #converting initial frequency into initial energy value which will be conserved
        return [X,dX,Y,dY,E,alpha]

    def wave_position(self, wl):
        return np.clip((wl - 330) / 560,0,1) #min lambda = 330, delta lambda = 560

    def find_linger(self): #determines how long a frame can last
        full_freq = np.arange(self.min_frequency,self.max_frequency+1,self.angular_resolution)
        heights = np.ones(len(full_freq)) * self.max_H #self.
        Full_Frame = [self.createFrameC(full_freq,heights)] #s
        index = 1
        while not ((np.abs(Full_Frame[0][0]) > self.xlim) + (np.abs(Full_Frame[0][2]) > self.ylim)).all():
            Full_Frame[0][0] += Full_Frame[0][1]
            Full_Frame[0][2] += Full_Frame[0][3]
            index += 1

        return index


    def colormapper(self, lam): ######################################################################################
        if self.colormap == 'rainbow': #we can just use the lambda values generated
            return self.w2r[np.abs(self.w2r[:,0] - lam).argmin()][1:]
        else: #must map lambda values to something for matplotlib to use
            return  matplotlib.cm.get_cmap(self.colormap)(self.wave_position(lam))



    def render(self, resume_from_frame = 0):
        
        samplerate, data = wavfile.read(self.filepath) #loading wavfile data
        data = data.T[0]


        N = samplerate // self.samplerate

        w = blackman(N)#Window to help clean up spectral leakage (i.e. make points more pointy)
        wl = blackman(len(data) % N) #for the last window
        self.xf = fftfreq(N, 1/samplerate)[:N//2]

        i = 0
        maxF = 0 #getting min/maxs from data
        minF = 1e9
        while N*i < len(data): #Determining min/max frequency of data / volume peak
            if N*(i+1) < len(data):
                yf = fft(data[N*i:N*(i+1)]* (w))#taking fft
            elif N*(i+1) > len(data):
                yf = fft(data[N*i:N*(i+1)]* (wl))
            y =  2.0/N * np.abs(yf[0:N//2]) #converting fft to real values
            y[y<5e-5] = 0 #murdering noise under a threshold
            indii = find_peaks(y)[0]
            i = i + 1
            if len(self.xf[indii]) > 0:
                maxF = max(self.xf[indii].max(),maxF) #storing highest frequency
                minF = min(self.xf[indii].min(), minF) #storing lowest frequency
                h = (2.0/N * np.abs((y[indii])[0:N//2])).max()
                self.max_H = max(self.max_H,h)

        self.max_frequency = min(self.max_frequency, maxF) #Imposing limits on bounds
        self.min_frequency = max(self.min_frequency, minF)
        self.max_frequency = self.max_frequency - self.max_frequency % self.angular_resolution #applying angular resolution restriction
        self.min_frequency = self.min_frequency - self.min_frequency % self.angular_resolution
        self.max_FI = 1 / self.max_frequency #inverse of maximum frequency

        frameWindow = self.find_linger() #how many active frames can exist at once
        rdr = self.dr / self.r

        if self.print_global_progress:
            print('Required Number of Frames: {:.0f}'.format(np.ceil(len(data) / N - 1)))

        i = 0
        Frames = [] #To keep track of current photons
        while N*i < len(data):
            freqs,heights = self.grabFreq(data[N*i:N*(i+1)], N) #grabbing frequencies
            Frames.append(self.createFrameC(freqs,heights)) #creating new active frame
            i +=1


        if self.print_global_progress:
            print('All Frame Data Gathered')
            print('Creating Animation Frames')

        try:
            os.makedirs(self.temp_dirs) #temporary directory for storing rendered frames
        except:
            pass

        filenames = [] #creating list of filenames to save temporary images as
        for i in range(len(Frames)):
            l0 = len(str(len(Frames))) - len(str(i)) #calulating how many leading 0's are needed
            frameNumber = l0 * '0' + str(i)
            filenames.append(self.temp_dirs+'/frame_' + frameNumber + '.jpg')

        #determining how many runs will be required to render all frames as determined by batchsize
        counts = len(Frames) // self.batchSize
        counts += ( counts*self.batchSize != len(Frames) ) #Adding extra count in the account that rounding error cuts us short


        if resume_from_frame: #checking to see if we start partway through
            resume_from_frame = resume_from_frame // self.batchSize

        for l in range(counts)[resume_from_frame:]:
            plots = []
            t = datetime.datetime.now()
            start = l*self.batchSize
            stop = min((l+1)*(self.batchSize),len(Frames))
            if self.print_local_progress:
                print('Current Batch: {} - {}'.format(start,stop-1))
                print('\tOrganizing Frames')

            for i in range(start, stop):
                lastFrame = max(i-frameWindow,0) #sets the oldest frame that will be rendered
                plotF = []
                for j in range(i,lastFrame-1,-1): #cycling through frames
                    for k in range(len(Frames[j][0])):
                        # (i - j + lastFrame) = how many steps from the initial starting position the line is drawn
                        xs = [Frames[j][0][k] + Frames[j][1][k] * (i - j),
                              Frames[j][0][k] + Frames[j][1][k] * (i - j + 1)] #confusing
                        ys = [Frames[j][2][k] + Frames[j][3][k] * (i - j),
                              Frames[j][2][k] + Frames[j][3][k] * (i - j + 1)]

                        if (np.abs(xs[0]) < self.xlim) and (np.abs(ys[0]) < self.ylim): #only saving points that will be within graph bounds
                            #getting scaled distance from origin
                            dist = 1 + rdr * (i - j + lastFrame)
                            wnI = self.calcWI(Frames[j][4][k], dist)
                            lam = wnI * self.c
                            plotF.append([np.array([xs,ys]),self.colormapper(lam),Frames[j][5][k]])
                if len(plotF) < 1:
                    plotF.append([np.array([[0,0],[0,0]]),"black", 1])
                plots.append(plotF)

            if self.print_local_progress:
                print('\tDrawing Frames')

            poolPlotter(self.cores,filenames[start:stop],plots)
            dt = datetime.datetime.now()

            if self.print_local_progress:
                print('Elapsed time: ' + str(dt - t))
        ########################################## End for-loop #############################################

        if self.print_global_progress:
            print('All Frames Drawn, Creating animation')

        try:
            ffmpeg.input('./' + self.temp_dirs + '/frame_%0'+ str(len(str(len(Frames))))+'d.jpg', framerate = self.samplerate,) \
                .output(self.temp_output, vf= "pad=ceil(iw/2)*2:ceil(ih/2)*2") \
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

            input_video = ffmpeg.input(self.temp_output)

            input_audio = ffmpeg.input(self.filepath)

            ffmpeg.concat(input_video, input_audio, v=1, a=1) \
                .output(self.output) \
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

            try:
                os.remove(self.temp_output)
            except:
                pass

        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e

        if self.remove_temp_dirs:
            if self.print_global_progress():
                print('Animation finished, deleting leftover data')
            for i in range(len(filenames)): #Removing files from temporary directory
                if os.path.isfile(filenames[i]):
                    os.remove(filenames[i])
            try:
                os.rmdir(dirs) #deleting temporary directory if empty
            except:
                pass
        else:
            if self.print_global_progress:
                print('Animation finished')

        return


    def show_mapping_functions(self, function = "all", individual = False, flip = False, figsize=(6,6)):
        #Plots examples of all saved mapping functions
        #Plots all functions by default, can be specified to plot 1
        #Plots all maps on one plot by default
        #Set flip to true to invert mapping

        xp = np.linspace(0,1,500)
        if function == "all": #plotting all maps
            funcs = list(self.func_dict.keys())
        else: #plotting one map
            funcs = [function]

        fig = plt.figure(figsize=figsize)
        for f in funcs:
            if f != "flip":
                if flip:
                    y = self.func_dict['flip'](self.func_dict[f],xp)
                else:
                    y = self.func_dict[f](xp)
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


    def show_colormap(self, save = False):
        #Plots example colormap
        x_points = np.linspace(330,890,1236)
        for x in x_points: #plotting background
            plt.hlines(x,0,1,color=self.colormapper(x))

        rI = self.r-self.volume_offset_max
        l = self.min_frequency - (self.min_frequency % 20)
        h = self.max_frequency - (self.max_frequency % 20)
        full_freq = np.arange(l,h+1,20)
        f = (self.max_frequency - full_freq) * self.frequency_flip + full_freq * (not self.frequency_flip) #checking for freq. flip
        f = f % (self.max_frequency / self.colormap_periods) #checking for periods
        f = f * (not self.colormap_mirror) + np.abs(f - self.max_frequency/(self.colormap_periods*2)) * self.colormap_mirror #checking for mirror

        omega =  self.c/self.mapFreq(f,330+560*self.colormap_range[0], 890*self.colormap_range[1])#finding initial frequency for each mapped wavelength
        E = self.findE(omega)
        dist = rI
        wnI = self.calcWI(E, dist)

        lam = wnI * self.c
        lam[lam < x_points[0]] = x_points[0] * 0.995
        lam[lam > x_points[-1]] = x_points[-1] * 1.005
        xp = (full_freq*self.max_FI) % 1
        plt.plot(xp[xp.argsort()],lam[xp.argsort()],color='black',label= 'Minimum Redshift')

        ang = xp*2*np.pi
        a1 = np.cos(ang)
        a2 = np.sin(ang)
        md = 6/(np.vstack(np.abs([a1,a2])).T).max(axis=1)

        for i in range(len(E)):
            wnI[i] = self.calcWI(E[i], md[i])

        lam = wnI * self.c
        lam[lam < x_points[0]] = x_points[0] * 0.995
        lam[lam > x_points[-1]] = x_points[-1] * 1.005
        xp = (full_freq*self.max_FI) % 1

        plt.plot(xp[xp.argsort()],lam[xp.argsort()],linestyle='dashed',color='black', label='Maximum Redshift')
        plt.xlim(0,1)
        plt.ylim(x_points[0],x_points[-1])
        plt.xticks([]);
        plt.text(0.0,x_points[0]-0.15*x_points[0],"Low Frequency")
        plt.text(0.77,x_points[0]-0.15*x_points[0],"High Frequency")
        plt.ylabel('Output Color')
        plt.legend()
        plt.yticks([]);
        plt.title('Output Color Mapping');
        if save:
            plt.savefig(a.filepath.split('.')[0] + "_Colormap.png", format='png',bbox_inches='tight');
        plt.show();
        return

    def show_full_frame(self, window = "full", accuracy="normal", dr = None, alpha = 1, save = False):
        #Works but is really slow for some reason? Slightly concerning for actual code speed
        #Plots a fully populated frame
        #Accuracy can be 'quick', 'normal', or 'high'
        #Setting dr value takes priority over accuracy
        if dr == None:
            if accuracy == 'high':
                dr = self.dr / 2
            elif accuracy == 'normal':
                dr = self.dr
            else:
                dr = self.dr * 2

        rdr = dr/self.r
        maxFI = 1/self.max_frequency
        l = self.min_frequency - (self.min_frequency % self.angular_resolution)
        h = self.max_frequency - (self.max_frequency % self.angular_resolution)

        full_freq = np.arange(l,h+1,self.angular_resolution)


        if window == "reduced":
          sixth = full_freq.shape[0] // 16 #reducing full freq window to be -pi/8 : pi/8
          full_freq = np.hstack([full_freq[:sixth],full_freq[-sixth:]])

        heights = np.ones(len(full_freq)) #* maxH

        #Getting all possible active frequencies
        Full_Frame = [self.createFrameC(full_freq,heights)]
        while not ((np.abs(Full_Frame[0][0]) > self.xlim) + (np.abs(Full_Frame[0][2]) > self.ylim)).all():
            for i in range(len(Full_Frame)):
                Full_Frame[i][0] += Full_Frame[i][1]
                Full_Frame[i][2] += Full_Frame[i][3]
            Full_Frame.append(self.createFrameC(full_freq,heights))

        #Plotting frequencies
        fig = plt.figure(figsize=self.figsize)
        for i in range(len(Full_Frame)):#
            for j in range(len(Full_Frame[i][0])):#
                xs = [Full_Frame[i][0][j], Full_Frame[i][0][j]+ Full_Frame[i][1][j] ]
                ys = [Full_Frame[i][2][j], Full_Frame[i][2][j]+ Full_Frame[i][3][j] ]

                dist = 1 + rdr * (len(Full_Frame) - i)
                wnI = self.calcWI(Full_Frame[i][4][j], dist)
                lam = wnI * self.c
                plt.plot(xs,ys,color=self.colormapper(lam),alpha=alpha)

        plt.xlim(-self.xlim,self.xlim)
        plt.ylim(-self.ylim,self.ylim)
        plt.scatter([0],[0],color='black',linewidth=1)
        plt.axis('off');
        if save:
            plt.savefig(a.filepath.split('.')[0] + "_Full_Frame.png", format='png',bbox_inches='tight');
        plt.show();
        return



    def check_frame_pollution(self):
        #Checks the current threshold value for noise merc'n

        #Start is identical to render()
        samplerate, data = wavfile.read(self.filepath) #loading wavfile data
        data = data.T[0]

        N = samplerate // self.samplerate

        w = blackman(N)#Window to help clean up spectral leakage (i.e. make points more pointy)
        wl = blackman(len(data) % N) #for the last window
        self.xf = fftfreq(N, 1/samplerate)[:N//2]

        i = 0
        maxF = 0 #getting min/maxs from data
        minF = 1e9
        while N*i < len(data): #Determining min/max frequency of data / volume peak
            if N*(i+1) < len(data):
                yf = fft(data[N*i:N*(i+1)]* (w))#taking fft
            elif N*(i+1) > len(data):
                yf = fft(data[N*i:N*(i+1)]* (wl))
            y =  2.0/N * np.abs(yf[0:N//2]) #converting fft to real values
            y[y<5e-5] = 0 #murdering noise under a threshold
            indii = find_peaks(y)[0]
            i = i + 1
            if len(self.xf[indii]) > 0:
                maxF = max(self.xf[indii].max(),maxF) #storing highest frequency
                minF = min(self.xf[indii].min(), minF) #storing lowest frequency
                h = (2.0/N * np.abs((y[indii])[0:N//2])).max()
                self.max_H = max(self.max_H,h)

        self.max_frequency = min(self.max_frequency, maxF) #Imposing limits on bounds
        self.min_frequency = max(self.min_frequency, minF)
        self.max_frequency = self.max_frequency - self.max_frequency % self.angular_resolution #applying angular resolution restriction
        self.min_frequency = self.min_frequency - self.min_frequency % self.angular_resolution
        self.max_FI = 1 / self.max_frequency #inverse of maximum frequency

        frameWindow = self.find_linger() #how many active frames can exist at once
        rdr = self.dr / self.r

        i = 0
        Frames = [] #To keep track of current photons
        all_heights = [] #for histogram later
        while N*i < len(data):
            freqs,heights = self.grabFreq(data[N*i:N*(i+1)], N) #grabbing frequencies
            all_heights.append(heights)
            Frames.append(self.createFrameC(freqs,heights)) #creating new active frame
            i +=1

        #Seeing how many individual lines are in each frame
        line_counts = [] #line counts from each sample of song
        for i in range(len(Frames)):
          line_counts.append(Frames[i][0].shape[0])
        line_counts = np.array(line_counts)

        frame_lc = [] #line counts included in each frame
        for i in range(len(line_counts)):
          frame_lc.append(line_counts[max(0,i-frameWindow):i].sum())
        frame_lc = np.array(frame_lc)

        frame_max = frame_lc.argmax() #Most polluted frame
        frame_average = np.abs(frame_lc - frame_lc.mean()).argmin() #average frame

        ftp = [frame_max,frame_average] #frames to plot
        plots = []
        for i in ftp:
            lastFrame = max(i-frameWindow,0) #sets the oldest frame that will be rendered
            plotF = []
            for j in range(i,lastFrame-1,-1): #cycling through frames
                for k in range(len(Frames[j][0])):
                    # (i - j + lastFrame) = how many steps from the initial starting position the line is drawn
                    xs = [Frames[j][0][k] + Frames[j][1][k] * (i - j),
                          Frames[j][0][k] + Frames[j][1][k] * (i - j + 1)] #confusing
                    ys = [Frames[j][2][k] + Frames[j][3][k] * (i - j),
                          Frames[j][2][k] + Frames[j][3][k] * (i - j + 1)]

                    if (np.abs(xs[0]) < self.xlim) and (np.abs(ys[0]) < self.ylim): #only saving points that will be within graph bounds
                        #getting scaled distance from origin
                        dist = 1 + rdr * (i - j + lastFrame)
                        wnI = self.calcWI(Frames[j][4][k], dist)
                        lam = wnI * self.c
                        plotF.append([np.array([xs,ys]),self.colormapper(lam),Frames[j][5][k]])
            if len(plotF) < 1:
                plotF.append([np.array([[0,0],[0,0]]),"black", 1])
            plots.append(plotF)

        #Plotting Our Frames
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        for j in range(len(plots)):
          for i in range(len(plots[j])):
                ax[j].plot(plots[j][i][0][0],plots[j][i][0][1],color=plots[j][i][-2],alpha=plots[j][i][-1])

          ax[j].set_xlim(-6,6) #setting up axes and jawn
          ax[j].set_ylim(-6,6)
          ax[j].scatter([0],[0],color='black',linewidth=1)
          ax[j].set_xticks([])
          ax[j].set_yticks([]);

        ax[0].set_title("Most Polluted Frame")
        ax[1].set_title("Average Frame")
        plt.show()

        avg_f_heights = np.hstack(all_heights[max(frame_average-41,0):frame_average])
        max_f_heights = np.hstack(all_heights[max(frame_max-41,0):frame_max])

        return max_f_heights[max_f_heights.argsort()], avg_f_heights[avg_f_heights.argsort()]

    def optimize_threshold(self, use_max = True, thresh_percentage = None, max_lines = 1000):
        print(self.min_Volume)
        try:
          print("Original Threshold")
          print(self.min_Volume)
          output = self.check_frame_pollution()
          if thresh_percentage >=0 and thresh_percentage <= 1:
            if use_max:
              # self.min_Volume = output[0][0] * thresh_percentage
              self.min_Volume = output[0][int(len(output[0]) * thresh_percentage)]
            else:
              # self.min_Volume = output[1][0] * thresh_percentage
              self.min_Volume = output[1][int(len(output[1]) * thresh_percentage)]
          else:
            if use_max:
              self.min_Volume = output[0][max_lines]
            else:
              self.min_Volume = output[1][max_lines]
        except:
          print("Error, Current Threshold is Smaller than new Threshold. Please reset threshold")
          return


        print("Modified Threshold")
        print(self.min_Volume)
        output = self.check_frame_pollution()
        return output


    #Output to-string for convenience
    def __str__(self):
        out =  "Input Audio File   :   " + self.filepath + "\n"
        out += "Output Mp4 File    :   " + self.output + "\n"
        out += "Output Framerate   :   {}\n".format(self.samplerate)
        out += "Colormap           :   " + self.colormap + "\n"
        out += "Cores using        :   {}\n".format(self.cores)
        return out

    #More exhaustive output string
    def details(self):
        out =  "Input Audio File      :   " + self.filepath + "\n"
        out += "Output Mp4 File       :   " + self.output + "\n"
        out += "Output Framerate      :   {}\n".format(self.samplerate)
        out += "Cores using           :   {}\n".format(self.cores)
        out += "Batchsize             :   {}\n\n".format(self.batchSize)

        out += "Colormap              :   " + self.colormap + "\n"
        out += "Colormap Range        :   [{},{}]\n".format(self.colormap_range[0], self.colormap_range[1])
        out += "Colormap Periods      :   {}\n".format(self.colormap_periods)
        out += "Colormap Mirror       :   " + str(self.colormap_mirror) + "\n\n"

        out += "Frequency Mapping     :   " + self.frequency_mapping + "\n"
        out += "Flip Frequency Map    :   " + str(self.frequency_flip) + "\n"
        out += "Minimum Frequency     :   {}\n".format(self.min_frequency)
        out += "Maximum Frequency     :   {}\n".format(self.max_frequency)
        out += "Minimum Volume        :   {}\n".format(self.min_Volume) + "\n"
        out += "Volume Offset         :   " + str(self.volume_offset) + "\n"
        out += "Volume Offset Max     :   {}\n".format(self.volume_offset_max)
        out += "Volume Alpha          :   " + str(self.volume_alpha) + "\n"
        out += "Volume Alpha Range    :   [{},{}]\n".format(self.volume_alpha_range[0],self.volume_alpha_range[1])
        out += "Volume Alpha Mapping  :   " + self.volume_alpha_mapping + "\n\n"

        out += "Starting radius       :   {:.8f}\n".format(self.r)
        out += "Resolution            :   {:.8f}\n".format(self.dr)
        out += "Rate of Redshift      :   {:.8f}\n\n".format(self.M)

        out += "Figure Size           :   ({},{})\n".format(self.figsize[0], self.figsize[1])
        out += "Figure X limit        :   {:.4f}\n".format(self.xlim)
        out += "Figure Y limit        :   {:.4f}\n".format(self.ylim)

        out += "Print Overall Progress:   " + str(self.print_global_progress) + "\n"
        out += "Print Render Progress :   " + str(self.print_local_progress) + "\n"
        out += "Temporary Directory   :   " + self.temp_dirs + "\n"
        out += "Delete Temp. Dir.     :   " + str(self.remove_temp_dirs) + "\n"

        print(out)
