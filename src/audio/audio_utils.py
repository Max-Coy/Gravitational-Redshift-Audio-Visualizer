from pathlib import Path
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
from typing import Tuple

def load_audio(filepath: str) -> Tuple[int, np.ndarray]:
    """
    Load an audio file into a raw waveform array.

    Supports:
        - .wav 

    Returns:
        samplerate (int)
        data (np.ndarray): mono waveform (float/int depending on source)
    """

    samplerate, data = wavfile.read(filepath)

    # Convert stereo → mono if needed
    if data.ndim > 1:
        data = data.T[0]

    return samplerate, data


def mp3_to_wav(input_path: str | Path) -> Path:
    """
    Convert an MP3 file to WAV format.

    Parameters
    ----------
    input_path : str | Path
        Path to input MP3 file.

    Returns
    -------
    Path
        Path to generated WAV file.
    """
    input_path = Path(input_path)

    if input_path.suffix.lower() != ".mp3":
        raise ValueError(f"Expected .mp3 file, got: {input_path.suffix}")

    print("MP3 detected, converting to WAV...")

    try:
        output_path = input_path.with_suffix(".wav")

        sound = AudioSegment.from_mp3(str(input_path))
        sound.export(str(output_path), format="wav")

        print("Conversion successful")

        return output_path

    except Exception as e:
        print(f"Audio conversion failed: {e}")
        raise