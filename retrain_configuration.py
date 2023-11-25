import matplotlib.pyplot as plt
import numpy as np
import numba
import os
import glob
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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def FIT_pruning(model, importance_scores, percent, bake=False):
    importance_dict = {}
    i = 0
    parameters_to_prune = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and (isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d))) :
            importance_dict[(m, 'weight')] = importance_scores[i]
            i += 1
            parameters_to_prune.append((m, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percent,
        importance_scores = importance_dict
    )
    if bake:
        for (m, n) in parameters_to_prune:
            prune.remove(m, n)
    return parameters_to_prune

def remove_pruning(model, parameters_to_prune):
    for (m, n) in parameters_to_prune:
        A = getattr(m, "weight_mask")
        setattr(m, 'weight_mask', torch.ones_like(A).to(device))

    for (m, n) in parameters_to_prune:
        prune.remove(m, n)


def parse_quantization_config(quant_config):
	k = quant_config
	
	parsed_quant_config = [k[0],[[k[1],k[2]],[k[3],k[4]]],[[k[5],k[6],k[7]],[k[8],k[9]]],[[k[10],k[11],k[12]],[k[13],k[14]]],[[k[15],k[16],k[17]],[k[18],k[19]]],k[20]]
	return parsed_quant_config

def retrain_configurations(config_directory, fp_ckpt, data_directory, batch_size, epochs, learning_rate):

	# grab the full precision checkpoint:
	fpnn = torch.load(fp_ckpt)


	# grab the useful config filepaths from the directory
	configuration_filepaths = []
	for filepath in glob.iglob(f'{config_directory}/*.pt'):

		ckpt = torch.load(filepath)
		# perform sanity check to test whether the ckpt has already been configured and trained:

		starting_epoch = 0
		try:
			if 'fine_tuning' in ckpt:
				if len(ckpt['fine_tuning']['state_accumulator']) < epochs:
					configuration_filepaths.append(filepath)
					starting_epoch = len(ckpt['fine_tuning']['state_accumulator'])
					pass
				else:
					continue
			elif 'fin_val_accuracy' not in ckpt:
				configuration_filepaths.append(filepath)
				pass
		except:
			continue

		quant_config = parse_quantization_config(ckpt['quant_config'])
		importance_scores = ckpt['importance_scores']
		pruning_percentage = ckpt['pruning_percentage']

		print('Filepath: ', filepath)
		print('Quant Config: ', quant_config)
		print('Pruning Percentage: ', pruning_percentage)
		

		# generate the model from the given configuration file
		quant_model = models.QuantResNet(models.QuantBasicBlock, [2, 2, 2, 2], quant_config=quant_config)
		if starting_epoch != 0:
			quant_model.load_state_dict(ckpt['fine_tuning']['model_state_dict_fin_val'])
		else:
			quant_model.load_state_dict(fpnn['model_state_dict_fin_val'])
		
		paramsqp = FIT_pruning(quant_model, importance_scores, pruning_percentage)
		quant_model.to(device)
		
		quant_model.train()

		# grab the dataset
		train_loader, test_loader = train_utils.get_cifar10_loaders(data_directory,batch_size,4096)

		criterion = nn.CrossEntropyLoss()

		optimizer = optim.SGD(quant_model.parameters(), lr=learning_rate,
							  momentum=0.9, weight_decay=5e-4)
		if starting_epoch != 0:
			optimizer.load_state_dict(ckpt['fine_tuning']['optimizer_state_dict'])
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=starting_epoch)
		else:
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

		# evaluate at the beginning of training as a sanity check
		val_loss, val_accuracy = train_utils.evaluate(quant_model, device, test_loader, criterion, 0)

		# Train the model - collecting various statistics
		best_accuracy = 0
		state_accumulator = []
		model_state = {}
		for epoch in range(starting_epoch, epochs):
			print('-'*80)
			
			train_loss, train_accuracy = train_utils.train(quant_model, device, train_loader, criterion, optimizer, epoch)
			val_loss, val_accuracy = train_utils.evaluate(quant_model, device, test_loader, criterion, epoch)
			scheduler.step()
			
			if val_accuracy > best_accuracy:
					
				best_accuracy = val_accuracy

				model_state['model_state_dict_top_val'] = quant_model.state_dict()
				model_state['top_val_accuracy'] = best_accuracy
				
			lr = [ group['lr'] for group in optimizer.param_groups ][0]
			
			print(f'Learning rate: {lr}')
			
			state = [train_loss, train_accuracy, val_loss, val_accuracy, lr]
			
			state_accumulator.append(state)

			model_state['state_accumulator'] = state_accumulator
			
			remove_pruning(quant_model, paramsqp)
			model_state['model_state_dict_fin_val'] = quant_model.state_dict()
			paramsqp = FIT_pruning(quant_model, importance_scores, pruning_percentage)
			model_state['fin_train_accuracy'] = train_accuracy
			model_state['fin_val_accuracy'] = val_accuracy
			model_state['optimizer_state_dict'] = optimizer.state_dict()
			print('... saving ...')

			ckpt['fine_tuning'] = model_state

			torch.save(ckpt, filepath)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-directory', type=str, help='input configuration directory', required=True)
	parser.add_argument('--fp-ckpt', type=str, help='full precision model checkpoint', required=True)
	parser.add_argument('--data-directory', type=str, help='directory containing the data', required=True)
	parser.add_argument('--batch-size', type=int, default=60, help='Batch size')
	parser.add_argument('--epochs', type=int, default=60, help='Training Epochs')
	parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
	args = parser.parse_args()
	retrain_configurations(**vars(args))