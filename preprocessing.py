import pandas as pd
import numpy as np
from importlib import import_module
from typing import Dict, List, Union
from tqdm import tqdm
import os
import glob
from pydub import AudioSegment
import librosa


def pydub_to_np(audio: AudioSegment, target_sr=16000) -> tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    # audio = audio.set_frame_rate(req_frame_rate)
    # return np.array(audio.get_array_of_samples(), dtype=np.float32)/ (
    #         1 << (8 * audio.sample_width - 1)), audio.frame_rate

    raw_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    max_val = np.iinfo(audio.array_type).max 
    normalized_samples = raw_samples / max_val  

    original_sr = audio.frame_rate

    # Resample to the target sample rate using librosa
    if original_sr != target_sr:
        chunk_np_a = librosa.resample(normalized_samples, orig_sr=original_sr, target_sr=target_sr)
    else:
        chunk_np_a = normalized_samples

    return chunk_np_a

def preprocess_audio(audio_file, speaker_name='Barack_Obama', chunk_duration=10, format='.wav', save_chunks = False, out_dir = './chunks/', out_frame_rate=16000):
  
    print(audio_file)
    print(speaker_name)

    # create output directory. If output dircetory exists, remove all contents of the output directory
    if save_chunks:
      if not os.path.exists(out_dir): 
          os.makedirs(out_dir)

      elif len(os.listdir(out_dir)) != 0:             
          files = os.listdir(out_dir)
          print(files)
          for f in files:
              os.remove(os.path.join(out_dir,f))

    # read audio file
    audio = AudioSegment.from_file(audio_file, format=format)
    audio_data = audio.set_channels(1)
    # audio_data = audio.set_frame_rate(out_frame_rate)

    sr = audio_data.frame_rate

    print(sr)

    # totdal duration of the audio file in milliseconds
    total_duration = audio_data.duration_seconds * 1000

    print(total_duration)

    # Split the audio into one-minute chunks
    base_index = 0
    chunk_index = 1
    start_time = 0
    end_time = 0
    step_size = 5000

    audio_dict = {}

    while end_time <= total_duration:
      
        end_time = start_time + chunk_duration * 1000
        
        # print(start_time)
        # print(end_time)

        # Generate the output filename
        file_index = base_index + chunk_index

        out_filename = speaker_name + f"_{file_index}"

        # Extract the chunk from the audio and export as a single audio file
        chunk = audio_data[start_time:end_time]

        # skip if audio is only silence otherwise save into a dictionary
        # print(len(chunk))
        if not (is_silent(chunk) or len(chunk) < 5000):

          if save_chunks:
              output_file = os.path.join(out_dir, out_filename + '.flac')
              chunk.export(output_file, format="flac")
          
          # chunk_np = np.array(chunk.get_array_of_samples(), dtype=np.float32)
          chunk_np = pydub_to_np(chunk)
        
          audio_dict[out_filename] = chunk_np

        # Update the start time and chunk index for the next chunk

        # start_time = int(end_time/2)
        start_time = start_time + step_size
        chunk_index += 1

    return audio_dict


def is_silent(audio_segment):
    """
    Checks if an audio segment is just silence.

    Args:
      audio_segment: A pydub.AudioSegment object.

    Returns:
      True if the audio segment is just silence, False otherwise.
    """

    # Get the RMS amplitude of the audio segment.
    rms = audio_segment.rms

    # If the RMS amplitude is below a certain threshold, then the audio segment
    # is considered to be silent.
    silence_threshold = 0  # dB

    return rms < silence_threshold