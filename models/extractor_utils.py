import numpy as np
from librosa.feature import melspectrogram
from librosa import load
from typing import Union, List, Dict, Optional
import pickle
import warnings
from pathlib import Path
import os

class SaveOutput:
	def __init__(self, avg_type='avg', randnetw=False):
		self.outputs = []
		self.activations = {}  # create a dict with module name
		self.detached_activations = None
		self.avg_type = avg_type
		self.randnetw = randnetw
	
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
			# print(f'Shape {k}: {v[0].detach().numpy().shape}')
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
				# detached_activations[f'{k}--batch'] = avg_activations_batch
			
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
		
		if self.randnetw:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
		else:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(self.detached_activations, f)
