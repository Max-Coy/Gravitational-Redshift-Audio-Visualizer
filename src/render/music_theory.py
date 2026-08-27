import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..physics.redshift_model import freq_to_angle
from ..core.config import RedshiftConfig

DATA_DIR = Path(__file__).parent

note_freqs = np.genfromtxt(
    DATA_DIR / "note_frequencies.txt",
    delimiter=",",
    dtype="str",
    comments=None
)

major_keys = np.genfromtxt(
    DATA_DIR / "major_keys.txt",
    delimiter=",",
    dtype="str",
    comments=None
)

minor_keys = np.genfromtxt(
    DATA_DIR / "minor_keys.txt",
    delimiter=",",
    dtype="str",
    comments=None
)

note_lookup = {} # dictionary of all notes and their frequencies at 440 Hz starting with the 0th octave
for note in note_freqs:
  note_lookup[str(note[0])] = note[1:].astype(float)

keys = np.vstack([major_keys, minor_keys])
key_lookup = {} # dictionary of all keys and the notes in them, A -> A major, a -> a minor etc...
for key in keys:
    key_lookup[str(key[0])] = key[1:]

def check_key_signature(config):
    fig, ax = plt.subplots()
    draw_key_signature(config, ax)
    ax.set_xlim(-config.xlim, config.xlim)
    ax.set_ylim(-config.ylim, config.ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter([0], [0], color="black", linewidth=1)
    plt.show()


def draw_key_signature(config, ax):
   notes = get_note_frequencies(config)
   for note in notes:
      label = note[0]
      frequency = note[1]
      angle = freq_to_angle(frequency, config)
      line_mag = config.xlim + config.ylim # doesn't matter if the line is too long so no need to take proper magnitude
      x = np.cos(angle) 
      y = np.sin(angle) 
      ax.plot([0, x * line_mag], [0, y * line_mag], color = config.key_signature_color, alpha = config.key_signature_alpha, zorder = -1)
      if config.note_labels:
        rotation = np.degrees(angle)
        label_mag = np.sqrt(config.xlim**2 + config.ylim**2) * config.note_label_distance

        if rotation > 90 and rotation < 270:
            rotation -= 180
        elif rotation < -90:
            rotation += 180

        label_x = x * label_mag - np.sin(angle) * config.note_label_offset
        label_y = y * label_mag + np.cos(angle) * config.note_label_offset

        ax.text(label_x, 
                label_y, 
                label, 
                rotation = rotation,
                ha = 'center', 
                va = 'center', 
                size = config.note_label_size,
                color = config.key_signature_color)
                
   
def update_frequency_bounds(config: RedshiftConfig):
    notes = get_note_frequencies(config)
    max_frequency = 0
    min_frequency = config.max_frequency
    for note in notes:
        max_frequency = max(max_frequency, note[1])
        min_frequency = min(min_frequency, note[1])

    config.max_frequency = max_frequency * 1.02
    config.min_frequency = min_frequency * 0.998


def get_note_frequencies(config: RedshiftConfig):
    """
    Returns an array of all the notes with their given frequencies within the specified
    octave range
    """
    if config.note_octaves[0] < 0 or config.note_octaves[1] > 8:
       print("Error, please choose octaves within [0,8]")
       return
    
    output = []
    for note in config.notes:
        frequencies = note_lookup[note]
        for i in range(config.note_octaves[0], config.note_octaves[1] + 1):
           output.append([f"{note}{i}", frequencies[i]])

    return output

def set_notes_from_key_signature(config: RedshiftConfig):
   config.notes = key_lookup[config.key_signature]