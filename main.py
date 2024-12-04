import os
import soundfile as sf
import json

from preprocessing import preprocess_audio
from asd_models import ASD



def upload(url, data):

    # put data to s3 bucket
    # put information to database

    ###### the next few lines will be replaced by database and s3 bucket code ######
    # we can either save the audio file locally and then pass the audio filename and save path to the preprocessing block
    # OR, we can read the audio file directly and pass the audio data to the preprocesing block
    data_dir = './data'
    filename = 'Barack_Obama.wav'

    speaker_name = 'Barack_Obama_test'

    audio_file = os.path.join(data_dir, filename)

    # audio_data = sf.read(audio_file)

    ############# Preprocessing block starts here #####################################

    chunk_dir = 'chunks_Barack_Obama_test'
    audio_chunks_dict = preprocess_audio(audio_file, speaker_name=speaker_name, save_chunks=True, out_dir=chunk_dir, format='.wav')

    print(audio_chunks_dict)

    # audio_chunks_dict = {}

    # print(audio_chunks_dict)

    # format for audio_chunks_dict = {chunk_1: audio_data, chunk_2: audio_data, ...}


    ############## Save audio_chunks_dict to database and s3 bucket ###################



    ############### ASD Algorithm #####################################################

    asd_model_aasist = ASD(model_type='rawnet', generate_score_file=True)

    # asd_model_rawnet = ASD(model_type='rawnet', generate_score_file=True)

    # score_df_aasist = asd_model_aasist.run(audio_chunks_dict, use_saved_chunks=True, chunk_dir=chunk_dir, speaker_name=speaker_name)
    score_df_aasist = asd_model_aasist.run(audio_chunks_dict, speaker_name=speaker_name)

    # score_df_rawnet = asd_model_rawnet.run(audio_chunks_dict, use_saved_chunks=True, chunk_dir=chunk_dir, speaker_name=speaker_name)

    print(score_df_aasist)
    # print(score_df_rawnet)


############### main ################
if __name__ == "__main__":

    url = 'https://deepblue.lib.umich.edu/data/concern/data_sets/3r074v67t'
    data = []
    
    upload(url, data)

   