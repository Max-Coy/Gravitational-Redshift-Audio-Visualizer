# Gravitational Redshift Audio Visualizer

This project converts audio files into physics-inspired visualizations by mapping frequency content into a simulated gravitational field.

The system processes audio (MP3/WAV) by splitting it into time-based frames, performing an FFT on each segment, and mapping frequency components into a configurable wavelength space. These wavelengths are then emitted from a simulated black hole and propagated outward, undergoing a redshift transformation as a function of distance.

The resulting motion and spectral evolution are rendered as a sequence of frames and combined into a final video using FFmpeg, synchronized with the original audio.

## Features

- FFT-based audio decomposition per frame
- Configurable frequency → wavelength mapping functions
- Multiple redshift models (user-definable)
- Colormap support (matplotlib + custom palettes)
- Multi-core processing for frame generation
- Automated video assembly with audio synchronization

## TODO
 *  Maybe add graph lines to show where notes are (e.g. A4, B4 ...) could add configurations to show lines for 
certain key signatures, could have prebuilts or an option to parse something like 
[A, B#, Cb, D ... G]

 *  Add option for leading or trailing silence

 *  Some option to have the output rotate as the song progresses, either volume dependant or via set rate (or mix of both)

 *  make sure we are clearing folder for temp output before we start saving (and or making sure we don't have any extra files
 e.g. we make a render with 1000 frames, and then one with 500, making sure frames 501-1000 are not accidentally in second render)

 *  sleaker progress bar / update statements, maybe keep current form as explicit option but then set default to a regular progress bar

 *  Reimplement helper functions: check_frame_pollution, optimize_threshold, show_[full_frame, mapping_functions, colormap]

 *  Clean up refactoring, probably still a decent amount of bloat left behind by gpt

## AI Usage
This project was originally written as a singular script file between 2022-2023 completely by hand. It was later refactored into its current structure with the assistance of GPT-5.5 with the underlying logic unchanged. 

