import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def pad_point_from_tensor(tensor_list, sub_num, max_points):
    if tensor_list[0].ndim == 2: #torch 1.7+
    # if len(tensor_list[0].shape) == 2: # torch 1.1
        fake_batch_shape = [len(tensor_list)] + [max_points] + [tensor_list[0].shape[1]] 
        s, n, c = fake_batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(fake_batch_shape, dtype=dtype, device=device)
        # torch 1.7+
        mask = torch.ones((s, n), dtype=torch.bool, device=device)
        # torch 1.1
        # mask = torch.ones((s, n), dtype=torch.uint8, device=device) 
        for point, pad_point, m in zip(tensor_list, tensor, mask):
            pad_point[: point.shape[0], : point.shape[1]].copy_(point[:max_points])
            m[: point.shape[0]] = 0
    else:
        raise ValueError('not supported')
    return tensor, mask

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError(f"activation should be relu/gelu/leakyrelu, not {activation}.")