"""Get ``symbolic_profile()`` result with real execution"""
from typing import Callable

import matplotlib
import torch
from torch.autograd.profiler_util import _format_memory
from zoo import tm_models, tmm_models

from siu._subclasses import MetaTensor
from siu.fx import symbolic_profile, symbolic_trace
from siu.fx.node_util import compute_size_in_bytes
from siu.fx.passes.graph_profile import sim_env

matplotlib.use('agg')

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def run_forward(mod: torch.nn.Module, data_gen: Callable, num_steps: int, verbose: bool = False):
    torch.cuda.reset_peak_memory_stats()
    param_mem = sum([compute_size_in_bytes(p) for p in mod.parameters()])
    activation_mem = 0
    mod.cuda()
    mod.train()
    for n in range(num_steps):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inner_mem = torch.cuda.memory_allocated(device="cuda:0")
        data = data_gen('cuda:0')

        # If we need to dive deep into the memory usage by
        # inspecting `saved_tensor_hooks`

        # =====================================================
        if verbose and n == 0:
            hook = sim_env(mod)
            with hook:
                output = mod(data)
            print(f'Memory estimation by saved_tensor_hooks: {_format_memory(compute_size_in_bytes(hook.ctx))}')
            del hook
        # =====================================================
        else:
            output = mod(data)
        activation_mem += (torch.cuda.memory_allocated(device="cuda:0") - inner_mem) / num_steps
        del output
    return activation_mem, param_mem


def symbolic_run_forward(mod: torch.nn.Module, data_gen: Callable, verbose=False, bias_addition_split=False):
    sample = data_gen('cuda:0')
    gm = symbolic_trace(mod, meta_args={"x": sample}, bias_addition_split=bias_addition_split)
    gm = symbolic_profile(gm, sample, verbose=verbose)
    activation_mem = list(gm.graph.nodes)[-1].meta['info'].accumulate_size
    param_mem = sum([n.meta['info'].param_size for n in gm.graph.nodes])
    del gm
    return activation_mem, param_mem


def main(args):

    hist = {}

    for m in tqdm(tm_models):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mod = m()
        activation_mem, param_mem = run_forward(mod,
                                                lambda device: torch.rand(args.batch_size, 3, 224, 224, device=device),
                                                args.num_steps,
                                                verbose=args.verbose)
        sym_activation_mem, sym_param_mem = symbolic_run_forward(
            mod,
            lambda device: torch.rand(args.batch_size, 3, 224, 224, device=device),
            bias_addition_split=args.bias_addition_split)
        hist[m.__name__] = {
            'activation_mem': activation_mem,
            'param_mem': param_mem,
            'sym_activation_mem': sym_activation_mem,
            'sym_param_mem': sym_param_mem,
        }
        mod.cpu()
        del mod
        if args.verbose:
            print(
                f'{m.__name__} \t| {_format_memory(activation_mem)} \t| {_format_memory(param_mem)} \t| {_format_memory(sym_activation_mem)} \t| {_format_memory(sym_param_mem)}'
            )

    plot_result(hist, args.img_dir + '/tm_models')

    hist = {}

    for m in tqdm(tmm_models):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mod = m()
        activation_mem, param_mem = run_forward(mod,
                                                lambda device: torch.rand(args.batch_size, 3, 224, 224, device=device),
                                                args.num_steps,
                                                verbose=args.verbose)
        sym_activation_mem, sym_param_mem = symbolic_run_forward(
            mod,
            lambda device: torch.rand(args.batch_size, 3, 224, 224, device=device),
            bias_addition_split=args.bias_addition_split)
        hist[m.__name__] = {
            'activation_mem': activation_mem,
            'param_mem': param_mem,
            'sym_activation_mem': sym_activation_mem,
            'sym_param_mem': sym_param_mem,
        }
        mod.cpu()
        del mod
        if args.verbose:
            print(
                f'{m.__name__} \t| {_format_memory(activation_mem)} \t| {_format_memory(param_mem)} \t| {_format_memory(sym_activation_mem)} \t| {_format_memory(sym_param_mem)}'
            )

    plot_result(hist, args.img_dir + '/tmm_models')


def plot_result(hist, img_dir=None):
    """
    Plot activation memory usage and parameter memory usage grouped by model names.
    """
    df = pd.DataFrame(hist).T

    # don't apply format memory
    df['activation_mem'] = df['activation_mem'].astype(int)
    df['param_mem'] = df['param_mem'].astype(int)
    df['sym_activation_mem'] = df['sym_activation_mem'].astype(int)
    df['sym_param_mem'] = df['sym_param_mem'].astype(int)

    df = df.reset_index()
    df = df.rename(columns={'index': 'model'})
    df = df.melt(id_vars=['model'], var_name='type', value_name='memory')
    df['type'] = df['type'].str.replace('_mem', '')
    df['type'] = df['type'].str.replace('sym_', 'symbolic ')
    df['type'] = df['type'].str.replace('activation', 'activation memory')
    df['type'] = df['type'].str.replace('param', 'parameter memory')

    # save plot only don't show
    if img_dir is not None:
        print(f'Saving plot to {img_dir}.png')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='memory', hue='type', data=df)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(img_dir + '.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='.', help='saved image directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_steps', type=int, default=5, help='number of steps')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--bias_addition_split', action='store_true', help='split bias addition')
    args = parser.parse_args()
    main(args)
