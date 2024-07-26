import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from importlib import import_module
from typing import Dict, List, Union
from torch import Tensor


class ASD():
    
    def __init__(self, model_type='aasist', config_path='./config/AASIST_ASVspoof5.conf', generate_score_file=False):

        self.model_type = model_type
        self.gen_score_file = generate_score_file

        if self.gen_score_file:

            self.save_path = save_path = './score_files/'

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        if self.model_type == 'aasist':

            with open('./config/AASIST_ASVspoof5.conf', "r") as f_json:
                config = json.loads(f_json.read())

            self.config = config
            self.model_config = config["model_config"]

            self.model = get_model(self.model_config)

    
    def run(self, audio_data: np.ndarray):

        score_df = self.produce_evaluation(data_dict=audio_data)

        return score_df

    
    def produce_evaluation(
            self,
            data_dict: Dict) -> None:

        """Perform evaluation and return a score dataframe"""

        model = self.model
        model.eval()

        fname_list = []
        score_list = []
        for utt_id, audio_data in tqdm(data_dict.items()):
            # batch_x = batch_x.to(device)

            X_pad = pad(audio_data, 64600)
            x_inp = Tensor(X_pad)

            x_inp = x_inp.unsqueeze(0).float()

            with torch.no_grad():
                _, audio_out = model(x_inp)
                audio_score = (audio_out[:, 1]).data.cpu().numpy().ravel()
            
            # add outputs
            fname_list.append(utt_id)
            score_list.extend(audio_score.tolist())

        # using pandas method
        score_dict = {'filename': fname_list, 'cm-score': score_list}

        score_df = pd.DataFrame(data=score_dict)

        if self.gen_score_file:
            score_df.to_csv(self.save_path, index=False, sep="\t")

            print("Scores saved to {}".format(self.save_path))

        return score_df


def get_model(model_config: Dict):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    # model = nn.DataParallel(_model(model_config))
    model = _model(model_config)
    # model = model.to(device)
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