from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import torchvision.transforms as transforms
import torchvision
import torch.nn.utils.prune as prune

import models as models
import train_utils as train_utils
import FIT_utils as FIT_utils
from brevitas import config as bconfig
bconfig.IGNORE_MISSING_KEYS = True
import random
import string
from torchsummary import summary

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def quant_accumulator(model, layer_filter=None):
	def layer_filt(nm):
		if layer_filter is not None:
			return layer_filter not in name
		else:
			return True
	data = []
	for name, module in model.named_modules():
		if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))) and (layer_filt(name)):
			for n, p in list(module.named_parameters()):
#                 print(n)
				if n.endswith('weight_orig') or n.endswith('weight'):
					
					if hasattr(module, 'quant_weight'):
						data.append((module.quant_weight().value))
						print('quant')
					elif hasattr(module, 'weight_orig'):
						data.append((module.weight))
					else:
						data.append(p.data)

				else:
					p.collect = False
			continue
		for p in list(module.parameters()):
			if p.requires_grad:
				p.collect = False

	return data

def compute_fake_FIT(params_before, params_after, EFw):
	
	FIT_contributions = 0
	for (theta, theta_, f) in zip(params_before, params_after, EFw):
		delta = torch.mean((theta.detach().cpu() - theta_.detach().cpu())**2).numpy()
		FIT_contributions += f*delta
		
	return FIT_contributions

def compute_fake_FIT_params(params_before, params_after, FeM):
	
	FIT_contributions = 0
	for (theta, theta_, f) in zip(params_before, params_after, FeM):
		delta = torch.sum(f*(theta.detach().cpu() - theta_.detach().cpu())**2).numpy()
		FIT_contributions += delta
		
	return FIT_contributions

def fake_quantization_layerwise(params, bit_width, index): # fake quantization
	fake_quantized_params = []
	for i, p in enumerate(params):
		if i == index:
			ma = torch.max(torch.abs(p))
			mi = -1*ma
			d = (ma)/(2**(bit_width-1)-1)
			pi = torch.round((p - mi)/d)
			pq = pi*d + mi
			fake_quantized_params.append(pq)
		else:
			fake_quantized_params.append(p)
	return fake_quantized_params

def mask_by_importance(params, importance_scores, percent): # fake pruning
	shapes = [dd.shape for dd in params]
	lens = [dd.numel() for dd in params]
	lens_cum = list(np.cumsum(lens))
	lens_cum.insert(0,0)
	
	qI = torch.cat([g2.view(-1) for g2 in importance_scores]).detach().cpu()
	qp = torch.cat([g2.view(-1) for g2 in params]).detach().cpu()
	k = int(percent*len(qp))
	topk, indices = torch.topk(-qI, k)
	qprs = torch.scatter(qp, -1, indices, 0.)
	dc = []
	for i in range(len(shapes)):
		dc.append(torch.reshape(qprs[lens_cum[i]:lens_cum[i+1]], shapes[i]))
	
	return dc

def assign_parameters(model, params):
	i = 0
	for name, module in model.named_modules():
		if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))):
			for n, p in list(module.named_parameters()):
#                
				if n.endswith('weight_orig') or n.endswith('weight'):
					p.data = nn.parameter.Parameter(params[i].to(device))
					p.collect = True
					i+=1
	print('new parameters assigned')

def generate_FIT_pruning_importance(params, FeM):
	
	importance_scores = []
	for (theta, f) in zip(params, FeM):
		importance = f*(theta.detach().cpu()**2)
		importance_scores.append(importance)
	
	return importance_scores

def compute_fake_active_bytes(params, quantization_config): ## effectively the heuristic we're gonna use to choose the next version
	active_bytes = 0
	uncompressed = 0
	
	for p, c in zip(params, quantization_config):
		non_zero = torch.sum(torch.where(torch.abs(p)<10e-8, 0, 1)).detach().cpu().numpy()
		active_bytes += non_zero*c/8
		

		uncompressed += (p.numel()*4)
			
	return active_bytes/uncompressed

def fake_quantization(params, quantization_config):
	fake_quantized_params = []
	for p, c in zip(params, quantization_config):
		ma = torch.max(torch.abs(p))
		mi = -1*ma
		d = (ma)/(2**(c-1)-1)
		pi = torch.round((p - mi)/d)
		pq = pi*d + mi
		fake_quantized_params.append(pq)
	return fake_quantized_params

def renorm_heuristic(importance_scores, current, goal):
	qI = torch.sum(torch.cat([g2.view(-1) for g2 in importance_scores])).detach().cpu().numpy()
	return np.sqrt(((np.abs(current-goal)**2)*qI))

class FITerative_compression():
	def __init__(self, model, dataloader, criterion, goal=0.05):
		
		print('FITerative compression initialised')
		
		self.model = model

		self.dataloader = dataloader
		self.criterion = criterion
		
		self.fit_computer = FIT_utils.FIT(self.model, device, input_spec=(3, 32, 32))
		self.fit_computer.EF(self.model, self.dataloader, self.criterion)
		theta_t = quant_accumulator(model)
		importance_scores = generate_FIT_pruning_importance(theta_t, self.fit_computer.FeM)
		
		self.goal=goal
		
		self.start = self.node(theta_t, [3 for i in range(len(theta_t))], self.fit_computer.FeM.copy(), 0., 0., [0 for i in range(len(theta_t)+1)])
		h = self.h(self.start)
		self.open_lst = [self.start]
		
		self.quant_schedule = [3,2,0]
		self.pruning_schedule = 1-np.logspace(0, -3, base=10, num=40)
	
	def heuristic_function(self, v):
		theta = v['theta']
		quant_config = v['quant_config']
		FeM = v['FeM']
		current_compression = compute_fake_active_bytes(theta, quant_config)
		importance_scores = generate_FIT_pruning_importance(theta, FeM)

		return renorm_heuristic(importance_scores, current_compression, self.goal)

	def approximate_heuristic_function(self, v, prev_FeM):
		theta = v['theta']
		quant_config = v['quant_config']
		current_compression = compute_fake_active_bytes(theta, quant_config)
		importance_scores = generate_FIT_pruning_importance(theta, prev_FeM)

		return renorm_heuristic(importance_scores, current_compression, self.goal)
	
	def h(self, v, prev_FeM=None):
		
		theta = v['theta']
		quant_config = v['quant_config']
		
		goal_percentage = compute_fake_active_bytes(theta, quant_config)
		if prev_FeM is not None:
			heuristic = self.approximate_heuristic_function(v, prev_FeM)
		else:
			heuristic = self.heuristic_function(v)
		
		v['fscore'] = v['gscore'] + heuristic
		v['compression'] = goal_percentage
		print(v['fscore'], heuristic, goal_percentage)
	
	def node(self, theta, quant_config, FeM, pruning_percentage, gscore, state, fscore=None, compression=None):
		key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
		nn = {'theta': theta, 
			  'quant_config': quant_config, 
			  'FeM': FeM, 
			  'pruning_percentage': pruning_percentage,
			  'gscore': gscore, 
			  'fscore': fscore,
			  'compression': compression,
			  'state':state,'key': key}
		return nn
	
	def astar(self, greedy=True, approximate=True):
		
		iterations = 0
		while len(self.open_lst) > 0 and iterations < 1000:
			print('Iteration', iterations)
			n = None
			
			# it will find a node with the lowest value of f() -
			for v in self.open_lst:
				if v['compression'] < self.goal:
					return v
				
				
				if n == None or v['fscore'] < n['fscore']:
					n = v;
			print(n['quant_config'], n['pruning_percentage'])
			if greedy:
				self.open_lst = [n]
			self.explore_neighbours(n, approximate=approximate)
			
			iterations += 1
	
	def explore_neighbours(self, current_node, approximate=False):

		# if approximate, update FeM for the best node:
		if approximate:
			assign_parameters(self.model, current_node['theta'])
			self.fit_computer.EF(self.model, self.dataloader, self.criterion, min_iterations=20, max_iterations=20)
			current_node['FeM'] = self.fit_computer.FeM.copy()

		
		# generate importance scores for the current node:
		importance_scores = generate_FIT_pruning_importance(current_node['theta'], current_node['FeM'])
		current_state = current_node['state'].copy()
#         print(current_node['state'])
#         print(current_node['fscore'])
#         print(current_node['compression'])
		
		

				
		# new nodes via quantization
		for i in range(len(current_state)-1):
			neighbour_state = current_state.copy()
			neighbour_state[i] += 1 # move to the next state
			
			neighbour_quant_config = current_node['quant_config'].copy()
			neighbour_quant_config[i] = self.quant_schedule[neighbour_state[i]]
			if neighbour_quant_config[i] == 0:
				continue
			neighbour_pruning_percentage = current_node['pruning_percentage']
			# now update the parameters and other relavant node info
			
			neighbour_theta = fake_quantization(current_node['theta'].copy(), neighbour_quant_config)
			neighbour_theta = mask_by_importance(neighbour_theta, importance_scores, neighbour_pruning_percentage)
			
			neighbour_gscore = current_node['gscore'] + np.sqrt(compute_fake_FIT_params(current_node['theta'], neighbour_theta, current_node['FeM']))


			# decide whether to use a prev1 estimate as our metric tensor for computing the fscore
			if approximate:
				neighbour_node = self.node(neighbour_theta, 
									   neighbour_quant_config, 
									   current_node['FeM'].copy(), 
									   neighbour_pruning_percentage, 
									   neighbour_gscore,
									   neighbour_state)
				_ = self.h(neighbour_node)
				self.open_lst.append(neighbour_node)
			else:
				assign_parameters(self.model, neighbour_theta)
				self.fit_computer.EF(self.model, self.dataloader, self.criterion, min_iterations=20, max_iterations=20) # recompute importance values and such
				neighbour_FeM = self.fit_computer.FeM.copy()
				
				neighbour_node = self.node(neighbour_theta, 
										   neighbour_quant_config, 
										   neighbour_FeM, 
										   neighbour_pruning_percentage, 
										   neighbour_gscore,
										   neighbour_state)
				
				_ = self.h(neighbour_node)
				self.open_lst.append(neighbour_node)
			
		
		#  new nodes via pruning:
		neighbour_state = current_state.copy()
		neighbour_state[-1] += 1 # move to the next state
		
		neighbour_quant_config = current_node['quant_config'].copy()
		neighbour_pruning_percentage = self.pruning_schedule[neighbour_state[-1]]
		
		# now update the parameters and other relavant node info
			
		neighbour_theta = fake_quantization(current_node['theta'].copy(), neighbour_quant_config)
		neighbour_theta = mask_by_importance(neighbour_theta, importance_scores, neighbour_pruning_percentage)

		neighbour_gscore = current_node['gscore'] + np.sqrt(compute_fake_FIT_params(current_node['theta'], neighbour_theta, current_node['FeM']))
		

		# decide whether to use a prev1 estimate as our metric tensor for computing the fscore
		
		if approximate:
			neighbour_node = self.node(neighbour_theta, 
								   neighbour_quant_config, 
								   current_node['FeM'].copy(), 
								   neighbour_pruning_percentage, 
								   neighbour_gscore,
								   neighbour_state)
			_ = self.h(neighbour_node)
			self.open_lst.append(neighbour_node)
		else:
			assign_parameters(self.model, neighbour_theta)
			self.fit_computer.EF(self.model, self.dataloader, self.criterion, min_iterations=20, max_iterations=20) # recompute importance values and such
			neighbour_FeM = self.fit_computer.FeM.copy()
			
			neighbour_node = self.node(neighbour_theta, 
									   neighbour_quant_config, 
									   neighbour_FeM, 
									   neighbour_pruning_percentage, 
									   neighbour_gscore,
									   neighbour_state)
			
			_ = self.h(neighbour_node)
			self.open_lst.append(neighbour_node)
		
		# remove previous node from the open set:
		current_key = current_node['key']
		for i, v in enumerate(self.open_lst):
			if v['key'] == current_key:
				del self.open_lst[i]

def generate_configurations(fp_ckpt, compression_range, data_directory, output_directory):
	# grab the full precision checkpoint:
	fpnn = torch.load(fp_ckpt)

	model = models.ResNet(models.BasicBlock, [2, 2, 2, 2])
	model.load_state_dict(fpnn['model_state_dict_fin_val'])
	model.to(device)

	# generate the desired compression shedule
	compression_rates = np.logspace(np.log10(compression_range[0]), np.log10(compression_range[1]), base=10, num=int(compression_range[2]))

	print('Target Compression Rates: ', compression_rates)

	# grab the data
	train_loader, test_loader = train_utils.get_cifar10_loaders(data_directory,60,4096)
	hess_loader, _ = train_utils.get_cifar10_loaders(data_directory,256,4096)

	# Run eval on the uncompressed model - for sanity check
	criterion = nn.CrossEntropyLoss().to(device)
	train_utils.evaluate(model, device, test_loader, criterion, 0)

	accumulate_configurations = []

	for compression_rate in compression_rates:
		c = FITerative_compression(model, dataloader=hess_loader, criterion=criterion, goal=compression_rate)
		optimal_configuration = c.astar()
		print(optimal_configuration['quant_config'], optimal_configuration['pruning_percentage'], optimal_configuration['compression'])
		optimal_configuration['importance_scores'] = generate_FIT_pruning_importance(optimal_configuration['theta'], optimal_configuration['FeM'])
		torch.save(optimal_configuration, f"{output_directory}compression_configuration_{optimal_configuration['compression']:.3f}ab_{np.mean(optimal_configuration['quant_config']):.1f}mp_{optimal_configuration['pruning_percentage']:.2%}p.pt")


	# save the generated configurations



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fp-ckpt', type=str, help='full precision model checkpoint', required=True)
	parser.add_argument('--compression-range', nargs='+', default=[0.25, 0.005, 12], type=float, help='(min value, max value, number)')
	parser.add_argument('--data-directory', type=str, help='directory containing the data', required=True)
	parser.add_argument('--output-directory', type=str, help='directory containing the data', required=True)
	args = parser.parse_args()
	generate_configurations(**vars(args))