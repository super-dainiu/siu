from typing import Callable, Dict, List, Optional, Tuple, Union

import torch


def compute_size_in_bytes(elem: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Compute the size of a tensor or a collection of tensors in bytes.

    Args:
        elem (Union[torch.Tensor, Dict, List, Tuple, int]): Arbitrary nested ``torch.Tensor`` data structure.

    Returns:
        int: The size of the tensor or the collection of tensors in bytes.
    """
    nbytes = 0
    if isinstance(elem, torch.Tensor):
        if elem.is_quantized:
            nbytes += elem.numel() * torch._empty_affine_quantized([], dtype=elem.dtype).element_size()
        else:
            nbytes += elem.numel() * torch.tensor([], dtype=elem.dtype).element_size()
    elif isinstance(elem, dict):
        value_list = [v for _, v in elem.items()]
        nbytes += compute_size_in_bytes(value_list)
    elif isinstance(elem, tuple) or isinstance(elem, list) or isinstance(elem, set):
        for e in elem:
            nbytes += compute_size_in_bytes(e)
    return nbytes
