"""Depreciated"""

import torch
from colossalai.auto_parallel.passes.runtime_apply_pass import (
    runtime_apply,
    runtime_apply_for_iterable_object,
    runtime_comm_spec_apply,
)
from colossalai.tensor.comm_spec import CommSpec

from siu._subclasses import MetaTensor
from siu.fx.passes.shape_prop import register_shape_impl


@register_shape_impl(runtime_apply)
def runtime_apply_impl(tensor, origin_dict, input_dict, node_index, user_node_index):
    out_shape = input_dict[node_index][user_node_index].get_sharded_shape_per_device()
    return MetaTensor(torch.empty(out_shape, device='meta', dtype=tensor.dtype), device=tensor.device)


@register_shape_impl(runtime_apply_for_iterable_object)
def runtime_apply_for_iterable_object_impl(tensor_list, origin_dict, input_dict, node_index, user_node_index):
    out_shape_list = [spec.get_sharded_shape_per_device() for spec in input_dict[node_index][user_node_index]]
    return [
        MetaTensor(torch.empty(out_shape, device='meta', dtype=tensor.dtype), device=tensor.device)
        for out_shape, tensor in zip(out_shape_list, tensor_list)
    ]


@register_shape_impl(runtime_comm_spec_apply)
def runtime_comm_spec_apply_impl(tensor, comm_actions_dict, node_index, op_data_name):
    comm_action = comm_actions_dict[node_index][op_data_name]
    if isinstance(comm_action.comm_spec, CommSpec):
        rst = comm_action.comm_spec.covert_spec_to_action(tensor)
    else:
        pass
