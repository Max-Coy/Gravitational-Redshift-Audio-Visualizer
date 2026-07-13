from pathlib import Path
import numpy as np
import os
import datetime

# Custom Function imports
from .core.config import RedshiftConfig
from .audio.audio_utils import load_audio, mp3_to_wav
from .audio.fft_pipeline import analyze_audio_bounds, grab_freq
from .render.renderer import build_render_context, build_plot_frame, pool_plotter, cleanup_render_directory
from .visualization.frames import build_frames
from .render.ffmpeg_utils import build_video

class RedshiftAnimation:
    def __init__(self, input_path, output_path = None, config = None):
        #Animation Settings
        self.config = config or RedshiftConfig()
        
        #Input / Output Files
        self.input_path = Path(input_path)
        if self.input_path.suffix.lower() == ".mp3":
            self.input_path = mp3_to_wav(self.input_path)

        if output_path is None:
            self.output_path = self.input_path.with_name(
                self.input_path.stem + "_redshift_animation.mp4"
            )
        else:
            self.output_path = Path(output_path)
        


        #Lookup table for wavelength to rgb conversion
        self.w2r = np.genfromtxt("src/w2r_blend.csv", delimiter=" ")

        # Runtime values
        self.max_FI = 1 / self.config.max_frequency
        self.min_Volume = 1e-10 # Minimum fft signal
        self.xf = None #x-values for fft
        self.max_H = 1 #maximum volume recorded
        self.redshift_rate = 1
        self.frame_window = 1


    def render(self, resume_from_frame = 0):
        if self.config.print_global_progress:
            print("Loading Data...")

        samplerate, data = load_audio(self.input_path)

        # (min_freq, max_freq, max_amp, fft_axis, N)
        analysis = analyze_audio_bounds(samplerate, data, self.config)
        self.max_H = analysis[2]

        (
            self.config.min_frequency, 
            self.config.max_frequency,
            self.redshift_rate,
            self.frame_window,
         ) = build_render_context(self.config, analysis)

        self.max_FI = 1 / self.config.max_frequency
        

        
        Frames = build_frames(data, analysis[4], analysis[3], self.max_H, self.config)
        
        os.makedirs(self.config.temp_dirs, exist_ok=True)

        filenames = [f"{self.config.temp_dirs}/frame_{i:0{len(str(len(Frames)))}d}.jpg"
        for i in range(len(Frames))
        ]

        num_frames = len(Frames)

        if self.config.print_global_progress:
            print('Required Number of Frames: {:.0f}'.format(np.ceil(len(data) / analysis[4] - 1)))
            print('Creating Animation Frames...')
        

        #determining how many runs will be required to render all frames as determined by batchsize
        counts = (num_frames + self.config.batch_size - 1) // self.config.batch_size

        start_batch = resume_from_frame // self.config.batch_size

        for batch_idx in range(start_batch, counts):

            plots = []
            t = datetime.datetime.now()

            start = batch_idx * self.config.batch_size
            stop = min((batch_idx + 1) * self.config.batch_size, num_frames)

            if self.config.print_local_progress:
                print(f"Current Batch: {start} - {stop - 1}")
                print("\tOrganizing Frames")

            for i in range(start, stop):
                plotF = build_plot_frame(
                    i,
                    Frames,
                    self.frame_window,
                    self.redshift_rate,
                    self.config,
                    self.w2r
                )
                
                plots.append(plotF)
        
            if self.config.print_local_progress:
                print('\tDrawing Frames')
            
            pool_plotter(self.config.cores, filenames[start:stop], plots, self.config)
            dt = datetime.datetime.now()

            if self.config.print_local_progress:
                print('Elapsed time: ' + str(dt - t))

        
        if self.config.print_global_progress:
            print('All Frames Drawn, Creating animation')

        build_video(self.config, self.input_path, self.output_path, len(Frames))

        cleanup_render_directory(Path(self.config.temp_dirs), self.config.remove_temp_dirs)

        if self.config.print_global_progress:
            print("All done :)")
