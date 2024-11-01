import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from importlib import import_module
from typing import Dict, List, Union
from torch import Tensor
import soundfile as sf
import librosa
import yaml

from models.RawNet import RawNet


class ASD():
    
    def __init__(self, model_type='aasist', generate_score_file=False, save_path = './score_files/'):

        self.model_type = model_type
        self.gen_score_file = generate_score_file

        #GPU device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
        print('Device: {}'.format(self.device))

        if self.gen_score_file:

            self.save_path = save_path

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        if self.model_type == 'aasist':

            config_path='./config/AASIST_ASVspoof5.conf'

            with open(config_path, "r") as f_json:
                config = json.loads(f_json.read())

            self.config = config
            self.model_config = config["model_config"]

            self.model = get_model(self.model_config, self.device)

            self.model.load_state_dict(torch.load(config["model_path"], map_location=self.device))

        if self.model_type == 'rawnet':

            dir_yaml = os.path.splitext('./config/model_config_RawNet')[0] + '.yaml'

            with open(dir_yaml, 'r') as f_yaml:
                parser1 = yaml.load(f_yaml, yaml.Loader)

                model = RawNet(parser1['model'], self.device)
                nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
                self.model =(model).to(self.device)
                print("no. model params:{}".format(nb_params))

            model_path = './models/weights/RawNet2/RawNet2_best_model_laundered_train.pth'

            if model_path:
                model.load_state_dict(torch.load(model_path,map_location=self.device))
                print('Model loaded : {}'.format(model_path))

    
    def run(self, audio_data_dict: Dict, use_saved_chunks = False, chunk_dir='./chunks', speaker_name='Barack_Obama'):

        if use_saved_chunks:

            chunk_dir = chunk_dir
            # chunk_dir = '/data/Famous_Figures/AES_Data/aes_data'

            chunk_files = os.listdir(chunk_dir)
            
            audio_data_dict = {}
            for cf in chunk_files:
                
                cf_filename = os.path.join(chunk_dir, cf)
                
                # audio_data, sr = sf.read(cf_filename)
                audio_data, fs = librosa.load(cf_filename, sr=16000)

                # audio_data = audio_data[:,0]

                audio_data_dict[cf.split('.')[0]] = audio_data

            score_df = self.produce_evaluation(data_dict=audio_data_dict, speaker_name=speaker_name)

        else:
            score_df = self.produce_evaluation(data_dict=audio_data_dict, speaker_name=speaker_name)

        return score_df

    
    def produce_evaluation(
            self,
            data_dict: Dict,
            speaker_name='Barack_Obama') -> None:

        """Perform evaluation and return a score dataframe"""

        model = self.model
        model.eval()

        fname_list = []
        score_list = []
        for utt_id, audio_data in tqdm(data_dict.items()):
            # batch_x = batch_x.to(device)

            X_pad = pad(audio_data, 64600)
            x_inp = Tensor(X_pad)

            x_inp = x_inp.unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                if self.model_type == 'aasist':
                    _, audio_out = model(x_inp)
                else:
                    audio_out = model(x_inp)

                audio_score = (audio_out[:, 1]).data.cpu().numpy().ravel()
                # print(audio_score)
            
            # add outputs
            fname_list.append(utt_id)
            score_list.extend(audio_score.tolist())

        # using pandas method
        score_dict = {'filename': fname_list, 'cm-score': score_list}

        score_df = pd.DataFrame(data=score_dict)

        if self.gen_score_file:

            score_file = os.path.join(self.save_path, speaker_name + '_' + self.model_type + '_out_scores.txt')
            score_df.to_csv(score_file, index=False, sep="\t")

            print("Scores saved to {}".format(score_file))

        return score_df


def get_model(model_config: Dict, device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    # model = nn.DataParallel(_model(model_config))
    model = _model(model_config)
    model = model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x