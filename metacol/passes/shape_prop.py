"""``torch.fx.ShapeProp``, but with ``MetaTensor``"""

from typing import Any, Tuple
from .._subclasses import MetaTensor


import torch
from torch.utils._pytree import tree_map

class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record the meta data of the result
    into the corresponding node.

    Usage:
        BATCH_SIZE = 2
        DIM_IN = 4
        DIM_HIDDEN = 16
        DIM_OUT = 16
        model = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, DIM_HIDDEN),
            torch.nn.Linear(DIM_HIDDEN, DIM_OUT),
            )
        input_sample = torch.rand(BATCH_SIZE, DIM_IN)
        gm = colossalai.fx.symbolic_trace(model)
        interp = ShapeProp(gm)
        interp.propagate(input_sample)

    Args:
         module (GraphModule): The module to be executed
    """
    
    def run_node(self, n: torch.fx.Node) -> Any:
        """
        Run a specific node ``n`` and return the result. Attach 'meta_data' to ``n``.

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        result = super().run_node(n)
        unwrap_fn = lambda x: x._tensor if isinstance(x, MetaTensor) else x
        n.meta['meta_data'] = tree_map(unwrap_fn, result)
        return result

    def propagate(self, *args, device=None):
        """
        Run `module` via interpretation and return the result and record the 
        shape of each node.
        Args:
            *args (Tensor): the sample input.
        Returns:
            Any: The value returned from executing the Module
        """
        wrap_fn = lambda x: MetaTensor(x, device=device)
        return super().run(*tree_map(wrap_fn, args))


def shape_prop_pass(gm: torch.fx.GraphModule, *args, device=None):
    ShapeProp(gm).propagate(*args, device=device)
    return gm