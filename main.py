import os
import soundfile as sf
import json

from preprocessing import preprocess_audio



def upload(url, data):

    print(data)

    # put data to s3 bucket
    # put information to database

    ###### the next few lines will be replaced by database and s3 bucket code ######
    # we can either save the audio file locally and then pass the audio filename and save path to the preprocessing block
    # OR, we can read the audio file directly and pass the audio data to the preprocesing block
    data_dir = './data'
    filename = 'A Promised Land - 004.wav'

    audio_file = os.path.join(data_dir, filename)

    # audio_data = sf.read(audio_file)

    ############# Preprocessing block starts here #####################################

    audio_chunks_dict = preprocess_audio(audio_file, save_chunks=True)

    print(audio_chunks_dict)

    # format for audio_chunks_dict = {chunk_1: audio_data, chunk_2: audio_data, ...}


    ############## Save audio_chunks_dict to database and s3 bucket ###################



    ############### ASD Algorithm #####################################################


############### main ################
if __name__ == "__main__":

    url = 'https://deepblue.lib.umich.edu/data/concern/data_sets/3r074v67t'
    data = []
    
    upload(url, data)

   