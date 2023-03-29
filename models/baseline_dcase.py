#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor

# add path above to import modules
import sys
sys.path.append('..')
from modules import Encoder
from modules import Decoder

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineDCASE']

## ADDED ##
import numpy as np
import os
import torch
from os.path import join, isfile
from os import listdir
from extractor_utils import *
from scipy.io import wavfile
import matplotlib.pyplot as plt

import random
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds/'
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/DCASE2020/'

files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
wav_files = [f for f in files if f.endswith('wav')]


class BaselineDCASE(Module):

    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 dropout_p_encoder: float,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int) \
            -> None:
        """Baseline method for audio captioning with Clotho dataset.

        :param input_dim_encoder: Input dimensionality of the encoder.
        :type input_dim_encoder: int
        :param hidden_dim_encoder: Hidden dimensionality of the encoder.
        :type hidden_dim_encoder: int
        :param output_dim_encoder: Output dimensionality of the encoder.
        :type output_dim_encoder: int
        :param dropout_p_encoder: Encoder RNN dropout.
        :type dropout_p_encoder: float
        :param output_dim_h_decoder: Hidden output dimensionality of the decoder.
        :type output_dim_h_decoder: int
        :param nb_classes: Amount of output classes.
        :type nb_classes: int
        :param dropout_p_decoder: Decoder RNN dropout.
        :type dropout_p_decoder: float
        :param max_out_t_steps: Maximum output time-steps of the decoder.
        :type max_out_t_steps: int
        """
        super().__init__()

        self.max_out_t_steps: int = max_out_t_steps

        self.encoder: Module = Encoder(
            input_dim=input_dim_encoder,
            hidden_dim=hidden_dim_encoder,
            output_dim=output_dim_encoder,
            dropout_p=dropout_p_encoder)

        self.decoder: Module = Decoder(
            input_dim=output_dim_encoder * 2,
            output_dim=output_dim_h_decoder,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the baseline method.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Predicted values.
        :rtype: torch.Tensor
        """
        h_encoder: Tensor = self.encoder(x)[:, -1, :].unsqueeze(1).expand(
            -1, self.max_out_t_steps, -1)
        return self.decoder(h_encoder)

def feature_extraction(audio_data: np.ndarray,
                       sr: int,
                       nb_fft: int,
                       hop_size: int,
                       nb_mels: int,
                       f_min: float,
                       f_max: float,
                       htk: bool,
                       power: float,
                       norm: bool,
                       window_function: str,
                       center: bool)\
        -> np.ndarray:
    """Feature extraction function.

    :param audio_data: Audio signal.
    :type audio_data: numpy.ndarray
    :param sr: Sampling frequency.
    :type sr: int
    :param nb_fft: Amount of FFT points.
    :type nb_fft: int
    :param hop_size: Hop size in samples.
    :type hop_size: int
    :param nb_mels: Amount of MEL bands.
    :type nb_mels: int
    :param f_min: Minimum frequency in Hertz for MEL band calculation.
    :type f_min: float
    :param f_max: Maximum frequency in Hertz for MEL band calculation.
    :type f_max: float|None
    :param htk: Use the HTK Toolbox formula instead of Auditory toolkit.
    :type htk: bool
    :param power: Power of the magnitude.
    :type power: float
    :param norm: Area normalization of MEL filters.
    :type norm: bool
    :param window_function: Window function.
    :type window_function: str
    :param center: Center the frame for FFT.
    :type center: bool
    :return: Log mel-bands energies of shape=(t, nb_mels)
    :rtype: numpy.ndarray
    """
    y = audio_data/abs(audio_data).max()
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T

    return np.log(mel_bands + np.finfo(float).eps)

def load_audio_file(audio_file: str, sr: int, mono: bool,
                    offset: Optional[float] = 0.0,
                    duration: Optional[Union[float, None]] = None)\
        -> np.ndarray:
    """Loads the data of an audio file.

    :param audio_file: The path of the audio file.
    :type audio_file: str
    :param sr: The sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: The audio data.
    :rtype: numpy.ndarray
    """
    return load(path=audio_file, sr=sr, mono=mono,
                offset=offset, duration=duration)[0]


if __name__ == '__main__':
    randnetw = True
    # default settings
    model = BaselineDCASE(input_dim_encoder=64, hidden_dim_encoder=256, output_dim_encoder=256, dropout_p_encoder=.25,
                          output_dim_h_decoder=256, nb_classes=4367, dropout_p_decoder=.25, max_out_t_steps=22)
    
    state_dict = torch.load('dcase_model_baseline_pre_trained.pt')

    if randnetw:
        print('OBS! RANDOM NETWORK!')

        ## The following code was used to generate indices for random permutation ##
        if not os.path.exists(os.path.join(os.getcwd(), 'DCASE2020_randnetw_indices.pkl')):
            d_rand_idx = {}  # create dict for storing the indices for random permutation
            for k, v in state_dict.items():
                w = state_dict[k]
                idx = torch.randperm(w.nelement())  # create random indices across all dimensions
                d_rand_idx[k] = idx

            with open(os.path.join(os.getcwd(), 'DCASE2020_randnetw_indices.pkl'), 'wb') as f:
                pickle.dump(d_rand_idx, f)

        else:
            d_rand_idx = pickle.load(open(os.path.join(os.getcwd(), 'DCASE2020_randnetw_indices.pkl'), 'rb'))

        for k, v in state_dict.items():
            w = state_dict[k]
            # Load random indices
            print(f'________ Loading random indices from permuted architecture for {k} ________')
            d_rand_idx = pickle.load(open(os.path.join(os.getcwd(), 'DCASE2020_randnetw_indices.pkl'), 'rb'))
            idx = d_rand_idx[k]
            rand_w = w.view(-1)[idx].view(w.size()) # permute, and reshape back to original shape
            state_dict[k] = rand_w
    
    model.load_state_dict(state_dict)   # map_location=torch.device('cpu'))
    model.eval()
    
    for file in wav_files:
        # write hooks for the model
        save_output = SaveOutput(avg_type='avg', randnetw=randnetw)
        
        hook_handles = []
        layer_names = []
        for idx, layer in enumerate(model.modules()):
            layer_names.append(layer)
            if isinstance(layer, torch.nn.GRU):
                print('Fetching GRU handles!\n')
                handle = layer.register_forward_hook(save_output)  # save idx and layer
                hook_handles.append(handle)
            if type(layer) == torch.nn.modules.Linear:
                print('Fetching ReLu handles!\n')
                handle = layer.register_forward_hook(save_output)  # save idx and layer
                hook_handles.append(handle)
        
        samplerate, data = wavfile.read(join(DATADIR, file))
    
        feats = feature_extraction(data, sr=44100,
                                   nb_fft=1024, hop_size=512, nb_mels=64, window_function='hann', center=True, f_min=.0,
                                   htk=False, power=1, norm=1, f_max=None)
        
        # add batch dim = 1 as third dim
        f = (torch.from_numpy(feats)).unsqueeze(0)
        model.forward(x=f)
        
        # detach activations
        detached_activations = save_output.detach_activations()
        
        # store and save activations
        # get identifier (sound file name)
        id1 = file.split('/')[-1]
        identifier = id1.split('.')[0]

        save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)
        
        
