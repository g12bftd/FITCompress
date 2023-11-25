import numpy as np
import numba
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import torchvision.transforms as transforms
import torchvision

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class FIT:
    def __init__(self, model, device, input_spec = (1,32,32), layer_filter=None):
        ''' Class for computing FIT
        Args:
            model
            device
            input_spec
            layer_filter - str - layers to ignore 
        '''
        
        names, param_nums, params = self.layer_accumulator(model, layer_filter)
        param_sizes = [p.size() for p in params]
        self.hook_layers(model, layer_filter)
        _ = model(torch.randn(input_spec)[None, ...].to(device))
        act_sizes = []
        act_nums = []
        for name, module in model.named_modules():
            if module.act_quant:
                act_sizes.append(module.act_in[0].size())
                act_nums.append(np.prod(np.array(module.act_in[0].size())[1:]))
                
        self.names = names
        self.param_nums = param_nums
        self.params = params
        self.param_sizes = param_sizes
        self.act_sizes = act_sizes
        self.act_nums = act_nums
        
        self.device = device
             
    def layer_accumulator(self, model, layer_filter=None):
        ''' Accumulates the required parameter information,
        Args:
            model
            layer_filter
        Returns:
            names - accumulated layer names
            param_nums - accumulated parameter numbers
            params - accumulated parameter values
        '''
        
        def layer_filt(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True
        layers = []
        names = []
        param_nums = []
        params = []
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))) and (layer_filt(name)):
                for n, p in list(module.named_parameters()):
                    if n.endswith('weight'):
                        names.append(name)
                        p.collect = True
                        layers.append(module)
                        param_nums.append(p.numel())
                        params.append(p)
                    else:
                        p.collect = False
                continue
            for p in list(module.parameters()):
                if p.requires_grad:
                    p.collect = False
#         print(len(layers))
#         print(np.sum(param_nums))
#         for i, (n, p) in enumerate(zip(names, param_nums)):
#             print(i, n, p)

        return names, np.array(param_nums), params
    
    def hook_layers(self, model, layer_filter=None):
        ''' Hooks the required activation information, which can be collected on network pass
        Args:
            model
            layer_filter
        '''

        def hook_inp(m, inp, out):
            m.act_in = inp

        def layer_filt(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True

        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))) and (layer_filt(name)):

                module.register_forward_hook(hook_inp)
                module.act_quant = True
            else:
                module.act_quant = False
                
    def EF(self, model, 
           data_loader, 
           criterion, 
           tol=1e-3, 
           min_iterations=100, 
           max_iterations=100):
        ''' Computes the EF
        Args:
            model
            data_loader
            tol - tolerance used for early stopping
            min_iterations - minimum number of iteration to include
            max_iterations - maximum number of iterations after which to break
        Returns:
            vFv_param_c - EF for the parameters
            vFv_act_c - EF for the activations
            F_param_acc - EF estimator accumulation for the parameters
            F_act_acc - EF estimator accumulation for the activations
            ranges_param_acc - parameter range accumulation
            ranges_act_acc - activation range accumulation
        '''
        
        model.eval()
        F_act_acc = []
        F_param_acc = []
        param_estimator_accumulation = []
        act_estimator_accumulation = []

        F_flag = False

        total_batches = 0.

        TFv_act = [torch.zeros(ps).to(self.device) for ps in self.act_sizes[1:]]  # accumulate result
        TFv_param = [torch.zeros(ps).to(self.device) for ps in self.param_sizes]  # accumulate result
        TFv_full = torch.zeros(np.sum(self.param_nums)).to(self.device) # accumulate result

        ranges_param_acc = []
        ranges_act_acc = []
        
        while(total_batches < max_iterations and not F_flag):
            
            for i, data in enumerate(data_loader, 1):
                model.zero_grad()
                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                batch_size = inputs.size(0)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                
                ranges_act = []
                actsH = []
                for name, module in model.named_modules():
                    if module.act_quant:
                        actsH.append(module.act_in[0])
                        ranges_act.append((torch.max(module.act_in[0]) - torch.min(module.act_in[0])).detach().cpu().numpy())

                ranges_param = []
                paramsH = []
                for paramH in model.parameters():
                    if not paramH.collect:
                        continue
                    paramsH.append(paramH)
                    ranges_param.append((torch.max(paramH.data) - torch.min(paramH.data)).detach().cpu().numpy())
                    
                G = torch.autograd.grad(loss, [*paramsH, *actsH[1:]])
                
                G2 = []
                for g in G:
                    G2.append(batch_size*g*g)
                fe = torch.cat([g2.view(-1) for g2 in G2])
                    
                indiv_param = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[:len(TFv_param)]])
                indiv_act = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[len(TFv_param):]])
                param_estimator_accumulation.append(indiv_param)
                act_estimator_accumulation.append(indiv_act)
                    
                TFv_param = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_param, G2[:len(TFv_param)])]
                ranges_param_acc.append(ranges_param)
                TFv_act = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_act, G2[len(TFv_param):])]
                ranges_act_acc.append(ranges_act)
                TFv_full = TFv_full + fe[:np.sum(self.param_nums)]
                
                total_batches += 1
                
                TFv_act_normed = [TFv_ / float(total_batches) for TFv_ in TFv_act]
                vFv_act = [torch.sum(x) for x in TFv_act_normed]
                vFv_act_c = np.array([i.detach().cpu().numpy() for i in vFv_act])

                TFv_param_normed = [TFv_ / float(total_batches) for TFv_ in TFv_param]
                vFv_param = [torch.sum(x) for x in TFv_param_normed]
                vFv_param_c = np.array([i.detach().cpu().numpy() for i in vFv_param])
                
                TFv_full_normed = TFv_full/float(total_batches)
                vFv_full_c = TFv_full_normed.detach().cpu().numpy()
                
                vFv_param_shape = [i.detach().cpu() for i in TFv_param_normed]
                
                F_act_acc.append(vFv_act_c)
                F_param_acc.append(vFv_param_c)
                
                if total_batches >= 2:
 
                    param_var = np.var((param_estimator_accumulation - vFv_param_c)/vFv_param_c)/total_batches
                    act_var= np.var((act_estimator_accumulation - vFv_act_c)/vFv_act_c)/total_batches
                    
#                     print(f'Iteration {total_batches}, Estimator variance: W:{param_var} / A:{act_var}')
                
                    if act_var < tol and param_var < tol and total_batches > min_iterations:
                        F_flag = True
                
                if F_flag or total_batches >= max_iterations:
                    break
        
        self.EFw = vFv_param_c
        self.EFa = vFv_act_c
        self.FAw = F_param_acc
        self.FAa = F_act_acc
        self.Rw = ranges_param_acc
        self.Ra = ranges_act_acc
        self.Fe = vFv_full_c
        self.FeM = vFv_param_shape
        
        return vFv_param_c, vFv_act_c, F_param_acc, F_act_acc, ranges_param_acc, ranges_act_acc
    
    # compute FIT:
    def noise_model(self, ranges, config):
        ''' Uniform noise model
        Args:
            ranges - data ranges
            config - bit configuration
        Returns:
            noise power
        '''
        return (ranges/(2**config - 1))**2

    def FITa(self, config):
        ''' computes FITa 
        Args:
            config - bit configuration for activations interlaced: [a1,a2,...]
        Returns:
            FIT value
        '''
        pert_acts = self.noise_model(np.mean(self.Ra, axis=0)[1:], config[1:])

        f_acts_T = pert_acts*self.EFa
        pert_T = np.sum(f_acts_T)
        return pert_T