import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import blackman


def analyze_audio_bounds(samplerate: int, data: np.ndarray, config) -> tuple[float, float, float]:
    """
    Extracts:
        - frequency bounds (min/max detected spectral peaks)
        - global amplitude envelope (max peak height)
        - FFT frequency axis
    """

    N = samplerate // config.samplerate

    window_main = blackman(N)
    window_tail = blackman(len(data) % N)

    fft_axis = fftfreq(N, 1 / samplerate)[: N // 2]

    max_freq = 0.0
    min_freq = float("inf")
    max_amp = 0.0
    i = 0
    while N * i < len(data):

        chunk = data[N * i : N * (i + 1)]
        window = window_main if len(chunk) == N else window_tail

        spectrum = fft(chunk * window)
        magnitude = 2.0 / N * np.abs(spectrum[: N // 2])

        # noise gate
        magnitude[magnitude < config.min_volume] = 0

        peaks = find_peaks(magnitude)[0]

        if len(peaks) > 0:
            freqs = fft_axis[peaks]

            max_freq = max(max_freq, freqs.max())
            min_freq = min(min_freq, freqs.min())

            mh = 2.0 / N * magnitude[peaks].max()

            max_amp = max(max_amp, mh)

        i += 1
    
    return min_freq, max_freq, max_amp, fft_axis, N

def grab_freq(
    data: np.ndarray,
    window: int,
    xf: np.ndarray,
    config,
):
    """
    Extract dominant frequencies and amplitudes from an audio window.

    Parameters
    ----------
    data : np.ndarray
        Audio samples for the current window.
    window : int
        FFT window size (usually sample chunk length).
    xf : np.ndarray
        Frequency axis (precomputed via fftfreq).
    config : RedshiftConfig
        Global configuration object.

    Returns
    -------
    freqs : np.ndarray
    heights : np.ndarray
    """

    # Window to help clean up spectral leakage (i.e. make points more pointy)
    wb = blackman(len(data)) 
    yf = fft(data * wb)
    y = 2.0 / window * np.abs(yf[0:window // 2])

    # noise thresholding
    y[y < 5e-5] = 0

    # Getting signal peaks
    indii = find_peaks(y)[0]
    if len(indii) == 0:
        return np.array([]), np.array([])
    
    # mapping signal peaks to angular resolution bins
    arr = np.array(
        list(
            set(
                xf[indii] - (xf[indii] % config.angular_resolution)
            )
        )
    )

    # Only taking signals above threshold strength
    height = (2.0 / window) * np.abs((y[indii])[0:window // 2])
    mask = height > config.min_volume

    return arr[mask], height[mask]


# def check_frame_pollution(self):
#     #Checks the current threshold value for noise merc'n

#     #Start is identical to render()
#     samplerate, data = wavfile.read(self.filepath) #loading wavfile data
#     data = data.T[0]

#     N = samplerate // self.samplerate

#     w = blackman(N)#Window to help clean up spectral leakage (i.e. make points more pointy)
#     wl = blackman(len(data) % N) #for the last window
#     self.xf = fftfreq(N, 1/samplerate)[:N//2]

#     i = 0
#     maxF = 0 #getting min/maxs from data
#     minF = 1e9
#     while N*i < len(data): #Determining min/max frequency of data / volume peak
#         if N*(i+1) < len(data):
#             yf = fft(data[N*i:N*(i+1)]* (w))#taking fft
#         elif N*(i+1) > len(data):
#             yf = fft(data[N*i:N*(i+1)]* (wl))
#         y =  2.0/N * np.abs(yf[0:N//2]) #converting fft to real values
#         y[y<5e-5] = 0 #murdering noise under a threshold
#         indii = find_peaks(y)[0]
#         i = i + 1
#         if len(self.xf[indii]) > 0:
#             maxF = max(self.xf[indii].max(),maxF) #storing highest frequency
#             minF = min(self.xf[indii].min(), minF) #storing lowest frequency
#             h = (2.0/N * np.abs((y[indii])[0:N//2])).max()
#             self.max_H = max(self.max_H,h)

#     self.max_frequency = min(self.max_frequency, maxF) #Imposing limits on bounds
#     self.min_frequency = max(self.min_frequency, minF)
#     self.max_frequency = self.max_frequency - self.max_frequency % self.angular_resolution #applying angular resolution restriction
#     self.min_frequency = self.min_frequency - self.min_frequency % self.angular_resolution
#     self.max_FI = 1 / self.max_frequency #inverse of maximum frequency

#     frameWindow = self.find_linger() #how many active frames can exist at once
#     rdr = self.dr / self.r

#     i = 0
#     Frames = [] #To keep track of current photons
#     all_heights = [] #for histogram later
#     while N*i < len(data):
#         freqs,heights = self.grabFreq(data[N*i:N*(i+1)], N) #grabbing frequencies
#         all_heights.append(heights)
#         Frames.append(self.createFrameC(freqs,heights)) #creating new active frame
#         i +=1

#     #Seeing how many individual lines are in each frame
#     line_counts = [] #line counts from each sample of song
#     for i in range(len(Frames)):
#         line_counts.append(Frames[i][0].shape[0])
#     line_counts = np.array(line_counts)

#     frame_lc = [] #line counts included in each frame
#     for i in range(len(line_counts)):
#         frame_lc.append(line_counts[max(0,i-frameWindow):i].sum())
#     frame_lc = np.array(frame_lc)

#     frame_max = frame_lc.argmax() #Most polluted frame
#     frame_average = np.abs(frame_lc - frame_lc.mean()).argmin() #average frame

#     ftp = [frame_max,frame_average] #frames to plot
#     plots = []
#     for i in ftp:
#         lastFrame = max(i-frameWindow,0) #sets the oldest frame that will be rendered
#         plotF = []
#         for j in range(i,lastFrame-1,-1): #cycling through frames
#             for k in range(len(Frames[j][0])):
#                 # (i - j + lastFrame) = how many steps from the initial starting position the line is drawn
#                 xs = [Frames[j][0][k] + Frames[j][1][k] * (i - j),
#                         Frames[j][0][k] + Frames[j][1][k] * (i - j + 1)] #confusing
#                 ys = [Frames[j][2][k] + Frames[j][3][k] * (i - j),
#                         Frames[j][2][k] + Frames[j][3][k] * (i - j + 1)]

#                 if (np.abs(xs[0]) < self.xlim) and (np.abs(ys[0]) < self.ylim): #only saving points that will be within graph bounds
#                     #getting scaled distance from origin
#                     dist = 1 + rdr * (i - j + lastFrame)
#                     wnI = self.calcWI(Frames[j][4][k], dist)
#                     lam = wnI * self.c
#                     plotF.append([np.array([xs,ys]),self.colormapper(lam),Frames[j][5][k]])
#         if len(plotF) < 1:
#             plotF.append([np.array([[0,0],[0,0]]),"black", 1])
#         plots.append(plotF)

#     #Plotting Our Frames
#     fig, ax = plt.subplots(1,2,figsize=(12,6))
#     for j in range(len(plots)):
#         for i in range(len(plots[j])):
#             ax[j].plot(plots[j][i][0][0],plots[j][i][0][1],color=plots[j][i][-2],alpha=plots[j][i][-1])

#         ax[j].set_xlim(-6,6) #setting up axes and jawn
#         ax[j].set_ylim(-6,6)
#         ax[j].scatter([0],[0],color='black',linewidth=1)
#         ax[j].set_xticks([])
#         ax[j].set_yticks([]);

#     ax[0].set_title("Most Polluted Frame")
#     ax[1].set_title("Average Frame")
#     plt.show()

#     avg_f_heights = np.hstack(all_heights[max(frame_average-41,0):frame_average])
#     max_f_heights = np.hstack(all_heights[max(frame_max-41,0):frame_max])

#     return max_f_heights[max_f_heights.argsort()], avg_f_heights[avg_f_heights.argsort()]


# def optimize_threshold(self, use_max = True, thresh_percentage = None, max_lines = 1000):
#     print(self.min_Volume)
#     try:
#         print("Original Threshold")
#         print(self.min_Volume)
#         output = self.check_frame_pollution()
#         if thresh_percentage >=0 and thresh_percentage <= 1:
#         if use_max:
#             # self.min_Volume = output[0][0] * thresh_percentage
#             self.min_Volume = output[0][int(len(output[0]) * thresh_percentage)]
#         else:
#             # self.min_Volume = output[1][0] * thresh_percentage
#             self.min_Volume = output[1][int(len(output[1]) * thresh_percentage)]
#         else:
#         if use_max:
#             self.min_Volume = output[0][max_lines]
#         else:
#             self.min_Volume = output[1][max_lines]
#     except:
#         print("Error, Current Threshold is Smaller than new Threshold. Please reset threshold")
#         return


#     print("Modified Threshold")
#     print(self.min_Volume)
#     output = self.check_frame_pollution()
#     return output