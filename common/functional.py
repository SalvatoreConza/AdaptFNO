from typing import Dict, Tuple, Any
import hashlib

import torch
import torch.nn.functional as F


def compute_velocity_field(input: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the velocity field along the a dimension of any input tensor

    Parameters:
        - tensor (torch.Tensor): Input tensor
        - dim (int): The axis along which the velocity field is computed

    Returns:
        torch.Tensor
    """
    output: torch.Tensor = (input ** 2).sum(dim=dim, keepdim=True) ** 0.5
    assert output.shape[dim] == 1
    return output


def hash_params(**kwargs) -> str:
    """
    Generates a unique hash string based on the provided keyword arguments.

    Args:
        **kwargs: Arbitrary keyword arguments that represent the parameters to hash.

    Returns:
        str: A unique MD5 hash string representing the input parameters.
    """
    # Sort kwargs by key to ensure consistent order
    d: Dict[str, Any] = {}
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (list, tuple)):
            d[key] = tuple(float(e) for e in value if isinstance(e, (int, float)))
        else:
            d[key] = value

    param_string: str = '_'.join([f"{key}:{value}" for key, value in d.items()])
    return hashlib.md5(param_string.encode()).hexdigest()

