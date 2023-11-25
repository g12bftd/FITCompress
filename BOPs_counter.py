import torch.nn as nn
import numpy as np
import torch

# Referece for hooks: https://github.com/sovrasov/flops-counter.pytorch

### Define the Hooks for counting flops: 

def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)
    
def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)
    
def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)
    
def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)

### Define the module reference:

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,
}

''' 
Computing the MACs/FLOPs for the network:
Computing the relative quantized bops provided the activation quantization remains the same
'''

def BOPs_metric(model, QT_depths=None, exclude={nn.ReLU, nn.Linear}, input_res = (3, 224, 224), verbose=False, layer_filter=None):
    ''' Computes the MACs for the Network and optionally a quantized network:

    Note that this is all entirely relative. MACs and 
    BOPs for each quantization level have a linear relation 
    *provided activations are held at a constant bit depth* 
    
    The calculation differs if we include activations! (TODO)

    Args:
        model: pytorch model object with compatible ops
        QT_depths: list of bit depths if quantization is desired
        exclude: Ops to exclude from the calculation
        input_res: tuple input resolution of the image
        verbose: if human-readable printout is desired
    Returns:
        bops: list of operations per layer
        QT_bops: list of operations for layer after quantization
    '''
    
    def layer_filt(nm):
        if layer_filter is not None:
            return layer_filter not in name
        else:
            return True
    
    layers = []
    names = []
    for name, module in model.named_modules():
        if type(module) in MODULES_MAPPING and (type(module) not in exclude) and layer_filt(name):
            names.append(name)
            layers.append(module)
            
    if verbose: print("Proceding to MACs calculation for {} blocks".format(len(layers)))
    
    for l in layers:
        l.__flops__ = 0
        l.__flops_handle__ = l.register_forward_hook(MODULES_MAPPING[type(l)])
        
    
    batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(model.parameters()).dtype,
                                             device=next(model.parameters()).device)
    
    _ = model(batch)

    def remove(layers):
        for l in layers:
            l.__flops_handle__.remove()
            del l.__flops_handle__
            del l.__flops__

    bops = [l.__flops__ for l in layers]
    if verbose:
        total = 0
        if QT_depths is None:
            cols = ["Layer", "MACs"]
            format_row = "{:>24}" * (len(cols))
            print('\n'+ format_row.format(*cols))
            print(len(cols)*24*'-')
            for l,n in zip(layers,names):
                total += l.__flops__
                row = [n, l.__flops__]
                print(format_row.format(*row))
            print('\n'+ f'Total MACs for chosen modules: {total}')
            remove(layers)
            return bops
        else:
            cols = ["Layer", "MACs", "Bits", "MACs"]
            format_row = "{:>24}" * (len(cols))
            print('\n'+ format_row.format(*cols))
            print(len(cols)*24*'-')
            total_Q = 0
            QT_bops = []
            for l,n,d in zip(layers, names, QT_depths):
                total += l.__flops__
                bf = int(l.__flops__*(d/32)**2)
                total_Q += bf
                QT_bops.append(bf)
                row = [n, l.__flops__, d, bf]
                print(format_row.format(*row))
            print('\n'+f'Total MACs for chosen modules: {total}')
            print(f'Total MACs for chosen Q modules: {total_Q}')
            remove(layers)
            return bops, QT_bops
    else:
        if QT_depths is None:
            remove(layers)
            return bops
        else:
            QT_bops = [int(l.__flops__*(d/32)**2) for l, d in zip(layers, QT_depths)]
            remove(layers)
            return bops, QT_bops