import dill
import numpy as np
from torch.autograd import Variable
import torch
from argparse import Namespace
import os
import sys

# path to sota-music-tagging-models needs to be appended
path_sota = '/home/verena/deployment/sota-music-tagging-models/'
sys.path.append(path_sota)
sys.path.append(os.path.join(path_sota, 'training'))
sys.path.append(os.path.join(path_sota, 'preprocessing'))
from preprocessing.mtat_read import Processor
from training.eval import Predict

# storing & loading explanations
def dill_dump(x, path):
    dill.dump(x, open(path, "wb"))


def dill_load(path):
    return dill.load(open(path, "rb"))


def composition_fn(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


# the following is required to work with https://github.com/minzwon/sota-music-tagging-models
tag_file = open(os.path.join(path_sota, "split", "msd", "50tagList.txt"), "r")
tags_msd = [t.replace('\n', '') for t in tag_file.readlines()]
path_models = os.path.join(path_sota, 'models')


won2020_default_config = Namespace()
won2020_default_config.num_workers = 10
won2020_default_config.dataset = 'msd'
won2020_default_config.model_type = 'musicnn'
won2020_default_config.batch_size = 16  # this is actually the nr. of chunks analysed per song
won2020_default_config.data_path = "placeholder"
won2020_default_config.load_data = True
won2020_default_config.input_length = 3 * 16000

audio_length_per_model = {
    'fcn': 29 * 16000,
    'crnn': 29 * 16000,
    'musicnn': 3 * 16000,
    'attention': 15 * 16000,
    'hcnn': 5 * 16000,
    'sample': 59049
}


def prepare_audio(audio_path, input_length, nr_chunks=None, return_snippet_starts=False):
    # based on code in Minz Won's repo
    if nr_chunks is None:
        nr_chunks = input_length
    processor = Processor()
    raw = processor.get_npy(audio_path)
    length = len(raw)
    snippet_starts = []

    hop = (length - input_length) // nr_chunks
    x = torch.zeros(nr_chunks, input_length)
    for i in range(nr_chunks):
        snippet_starts.append(i * hop)
        x[i] = torch.tensor(raw[i * hop:i * hop + input_length]).unsqueeze(0)
    if return_snippet_starts:
        return x, snippet_starts
    return x


def create_predict_fn(model, config):
    # this wrapper is used to created a configurable predict_fn that only takes x
    # and return y in the right format
    def predict_fn(x_array):
        x = torch.zeros(len(x_array), config.input_length)
        for i in range(len(x_array)):
            x[i] = torch.Tensor(x_array[i]).unsqueeze(0)
        x = x.cuda()
        x = Variable(x)
        y = model(x)
        y = y.detach().cpu().numpy()
        return np.array(y)
    return predict_fn


def get_model(config):
    model = Predict.get_model(config)

    if torch.cuda.is_available():
        S = torch.load(config.model_load_path)
    else:
        S = torch.load(config.model_load_path, map_location="cpu")

    model.load_state_dict(S)
    model.cuda()
    model.eval()
    return model