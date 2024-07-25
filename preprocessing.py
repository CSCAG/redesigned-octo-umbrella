import pandas as pd
import numpy as np
from importlib import import_module
from typing import Dict, List, Union
from tqdm import tqdm
import os

import whisper 
from stable_whisper import modify_model
from pydub import AudioSegment

def preprocess_audio(audio, speaker_name='Obama', chunk_len=10):

    print(audio)
    print(speaker_name)

    

    return 


def sentence_splitter(input_file,i, speaker_name):

    model = whisper.load_model('medium.en')

    # result1 = model.transcribe('/home/suryasss/Barack Obama_ Yes We Can.mp3', language='en', max_initial_timestamp=None)
    modify_model(model)
    result2 = model.transcribe(input_file, language='en') 
    a = result2.segments
    data = {
    'Segment no': range(0, len(a)),
    'start': [segment.start for segment in a],
    'end': [segment.end for segment in a],
    'text': [segment.text for segment in a]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df["end"] = df.end + 0.25
    df["total"] = df.end - df.start
    df['word_count'] = df['text'].apply(word_count)

    df = clupping(df)
    
    fil_df = df[df.total > 4][df.word_count >= 6].reset_index(drop = True)
    j = i+len(fil_df["end"])
    b = np.arange(i,j)
    fil_df["inde"] = b
    fil_df['filename'] = fil_df['inde'].apply(lambda x: f'{speaker_name}_{x:05d}.wav')
    

    for index, row in fil_df.iterrows():
        end_time = row['end']
        start_time = row['start']
        duration = end_time - start_time  # Calculate duration of the chunk
        output_file = os.path.join(out_dir, f"{config.speaker_name}_{i}.wav")  # Output file path for each chunk
        
        chunk =  AudioSegment.silent(duration = 100) + AudioSegment.from_file(input_file)[start_time * 1000:(end_time * 1000)] + AudioSegment.silent(duration = 100)

        chunk.export(output_file, format="wav")
        i = i+1    
    fil_df[["filename","text","start","end"]].to_csv(f'speaker_audios/{config.speaker_name}/meta_chunk.csv',index = False)
    os.remove(input_file)
    return i,fil_df[["filename","text","start","end"]]


def word_count(row):
    return len(row.split())


def clupping(a):
    grouped_segments = []
    current_group = []
    current_duration = 0.0
    a['start_diff'] = a['start'].diff().shift(-1)
    a['start_diff'] = a['start_diff'].fillna(a["total"])
    # Iterate over the dataframe rows
    for idx, row in a.iterrows():
        if current_duration < config.threshold :

            if current_duration + row['start_diff'] > config.threshold:
                # Combine the text of the current group
                combined_text = ' '.join([segment['text'] for segment in current_group])
                
                # Create a new row for the grouped segments
                new_row = {
                    'Segment no': [segment['Segment no'] for segment in current_group],
                    'start': current_group[0]['start'],
                    'end': current_group[-1]['end'],
                    'text': combined_text,
                    'total': current_duration,
                    'word_count': sum(segment['word_count'] for segment in current_group)
                }
                
                # Add the new row to the list of grouped segments
                grouped_segments.append(new_row)
                
                # Reset the current group and duration
                current_group = []
                current_duration = 0.0
        
        else:
                
                # Create a new row for the grouped segments
                new_row = {
                    'Segment no': row["Segment no"],
                    'start': row['start'],
                    'end': row['end'],
                    'text': row["text"],
                    'total': row["total"],
                    'word_count':  row["word_count"]
                }
                
                # Add the new row to the list of grouped segments
                grouped_segments.append(new_row)
                
                # Reset the current group and duration
                current_group = []
                current_duration = 0.0
        
        # Add the current row to the group
        current_group.append(row)
        current_duration += row['start_diff']

    # If there are any remaining rows in the current group, add them as a final segment
    if current_group:
        combined_text = ' '.join([segment['text'] for segment in current_group])
        new_row = {
            'Segment no': [segment['Segment no'] for segment in current_group],
            'start': current_group[0]['start'],
            'end': current_group[-1]['end'],
            'text': combined_text,
            'total': current_duration,
            'word_count': sum(segment['word_count'] for segment in current_group)
        }
        grouped_segments.append(new_row)

    # Convert the list of grouped segments to a new dataframe
    grouped_df = pd.DataFrame(grouped_segments)

    return grouped_df