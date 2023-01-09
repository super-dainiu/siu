"""Meta data function mappings.
func: (node) -> (saved_fwd_input, saved_fwd_buffer, saved_bwd_buffer, fwd_flops, bwd_flops, fwd_comm, bwd_comm)
"""
import torch

meta_info_fn = {}

def conv_meta_info(node: torch.fx.Node):
    return {
        "saved_fwd_input": [node.all_input_nodes]
    }