from dataclasses import dataclass


@dataclass
class RedshiftConfig:
    # Core physics
    M: float = 0.15 # Controls rate of redshifting (mass of center object)
    r: float = 2.0 # Starting radius of light beams
    dr: float = 0.2 # controls how long light beams are drawn (lower => slower light => higher resolution)

    # Audio / FFT
    samplerate: int = 24 # how often do we slice the audio file (FPS)
    angular_resolution: int = 40 # controls frequency bin size
    min_frequency: float = 0 # modifying will set a lower bound on plotted frequencies
    max_frequency: float = 3e5 # modifying will set an upper bound on plotted frequencies
    min_volume: float = 1e-10 
    

    # Rendering
    figsize: tuple[float, float] = (6,6) # dimensions of render in inches
    xlim: float = 6
    ylim: float = 6
    batch_size: int = 60 # sets how many frames will be rendered at once
    cores: int = 1 # sets how many cores will be used during rendering

    # Colormap
    colormap: str = "rainbow" #
    colormap_range: tuple[float, float] = (0.0, 1.0) # Specifies which part of colormap to use
    colormap_periods: int = 1 # setting to higher integers causes the colormap to repeat periodically
    colormap_mirror: bool = False # Mirror top and bottom half of color map

    # Visual Modulation
    frequency_mapping: str = "linear" # Controls how audio frequencies are mapped to wavelengths
    frequency_flip: bool = False # Flips high and low ends of frequency mapping

    volume_offset: bool = True # Stronger signals will be offset radially towards center
    volume_offset_max: float = 1.5 # Controls maximum volume offset (should be less than r)
    volume_offset_mapping: str = "4th_root" # mapping function for volume offset

    volume_alpha: bool = True # Weaker signals will have alpha channel value reduced
    volume_alpha_range: tuple[float, float] = (0.5, 1.0)
    volume_alpha_mapping: str = "sigmoid2"

    # IO / runtime
    temp_dirs: str = "temp_img_holder"
    remove_temp_dirs: bool = True
    print_global_progress: bool = True
    print_local_progress: bool = False

