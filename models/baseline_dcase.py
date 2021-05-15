#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module
from torch import Tensor

from modules import Encoder
from modules import Decoder

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineDCASE']


## ADDED ##
import numpy as np
from librosa.feature import melspectrogram
from librosa import load
from typing import Union, List, Dict, Optional
import pickle
import warnings
from pathlib import Path
import os
import librosa
import torch


class SaveOutput:
    def __init__(self, avg_type='avg'):
        self.outputs = []
        self.activations = {}  # create a dict with module name
        self.detached_activations = None
        self.avg_type = avg_type
    
    def __call__(self, module, module_in, module_out):
        """
		Module in has the input tensor, module out in after the layer of interest
		"""
        self.outputs.append(module_out)
        
        layer_name = self.define_layer_names(module)
        self.activations[layer_name] = module_out
    
    def define_layer_names(self, module):
        layer_name = str(module)
        current_layer_names = list(self.activations.keys())
        
        split_layer_names = [l.split('--') for l in current_layer_names]
        
        num_occurences = 0
        for s in split_layer_names:
            s = s[0]  # base name
            
            if layer_name == s:
                num_occurences += 1
        
        layer_name = str(module) + f'--{num_occurences}'
        
        if layer_name in self.activations:
            warnings.warn('Layer name already exists')
        
        return layer_name
    
    def clear(self):
        self.outputs = []
        self.activations = {}
    
    def get_existing_layer_names(self):
        for k in self.activations.keys():
            print(k)
        
        return list(self.activations.keys())
    
    def return_outputs(self):
        self.outputs.detach().numpy()
    
    def detach_one_activation(self, layer_name):
        return self.activations[layer_name].detach().numpy()
    
    def detach_activations(self):
        """
		Detach activations (from tensors to numpy)

		Arguments:

		Returns:
			detached_activations = for each layer, the flattened activations
			packaged_data = for LSTM layers, the packaged data
		"""
        detached_activations = {}
        
        for k, v in self.activations.items():
            # print(f'Shape {k}: {v.detach().numpy().shape}')
            print(f'Detaching activation for layer: {k}')
            if self.avg_type == 'avg_power':
                activations = activations ** 2
            
            if k.startswith('GRU'):
                activations = v
                # get both LSTM outputs
                activations_batch = activations[0].detach().numpy()
                activations_hidden = activations[1].detach().numpy()
    
                # squeeze batch dimension
                avg_activations_batch = activations_batch.squeeze()
                avg_activations_hidden = activations_hidden.squeeze()
    
                # CONCATENATE over the num directions dimension for hidden:
                avg_activations_hidden = avg_activations_hidden.reshape(-1)
                # mean over time
                avg_activations_batch = avg_activations_batch.mean(0)
    
                detached_activations[f'{k}--hidden'] = avg_activations_hidden
                detached_activations[f'{k}--batch'] = avg_activations_batch
            
            if k.startswith('Linear'):
                activations = v.detach().numpy().squeeze()
                actv_avg = np.mean(activations, axis=0)
                detached_activations[k] = actv_avg
        
        self.detached_activations = detached_activations
        
        return detached_activations
    
    def store_activations(self, RESULTDIR, identifier):
        RESULTDIR = (Path(RESULTDIR))
        
        if not (Path(RESULTDIR)).exists():
            os.makedirs((Path(RESULTDIR)))
        
        # filename = os.path.join(RESULTDIR, f'{identifier}_activations_inplaceReLUfalse.pkl')
        # filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
        filename = os.path.join(RESULTDIR, f'{identifier}_{self.avg_type}_activations.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(self.detached_activations, f)


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
# EOF

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
    model = BaselineDCASE(input_dim_encoder=64, hidden_dim_encoder=256, output_dim_encoder=256, dropout_p_encoder=.25,
                          output_dim_h_decoder=256, nb_classes=4367, dropout_p_decoder=.25, max_out_t_steps=22)


    model.load_state_dict(torch.load('dcase_model_baseline_pre_trained.pt'))   # map_location=torch.device('cpu'))
    model.eval()

    # write hooks for the model
    save_output = SaveOutput(avg_type='avg')

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



    file = '/Users/gt/Documents/GitHub/dcase-2020-baseline/data/clotho_audio_files/test/stim7_applause.wav'
    audio_input, _ = librosa.load(file, sr=44100)

    from scipy.io import wavfile

    samplerate, data = wavfile.read(file)

    feats = feature_extraction(data, sr=44100,
                               nb_fft=1024, hop_size=512, nb_mels=64, window_function='hann', center=True, f_min=.0,
                               htk=False, power=1, norm=1, f_max=None)

    f = (torch.from_numpy(feats)).unsqueeze(0)
    model.forward(x=f)
    
    # act_keys = list(save_output.activations.keys())
    # act_vals = save_output.activations
    
    # detach activations
    detached_activations = save_output.detach_activations()
    
    # model.forward(x=torch.from_numpy(feats))
    # model.forward(feats)
    #
    
