# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel._support.dtype import DataType
from typing import Any, Optional
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import get_default_scheduling_params
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave import allocate

from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randint,
    device_randn,
    device_randperm,
    device_zeros,
    to_default_device,
)
import torch
"""
General Questions:
TODO: Should arg names try to match torch as much as possible?
TODO: Should I make stride = 0 base stride for vector inserts?
TODO: Why can vector shapes be optional but when I don't put them it doesn't work? Also why does vector shapes as 1 work?
"""

def get_transponse_conv2d(
    layout: str, 
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    slice_stride_h: int,
    slice_stride_w: int,
    conv_stride: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol = GLOBAL_ADDRESS_SPACE,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    ratio_m: Optional[int] = None,
    ratio_n: Optional[int] = None,
) -> tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]:
    """This function creates a wave kernel for a transponse convolution in 2D also known as backwards convolution or deconvolution. This function takes advantage of the igemm stradegy to increase efficiency when computing the convolution.  
    
    Parameters:
        layout (str): Either nchw_fchw or nhwc_hwcf as the string for the format.
        n (int): Batch size, number of images per batch.
        h (int): Height of input matrices.
        w (int): Width of input matrices.
        c (int): Number of channels, ie 1 for greyscale, 3 for color.
        hf (int): Height of the filter.
        wf (int): Width of the filter.
        nf (int): Number of channels for filter (or total number of filters).
        slice_stride_h (int): Number of 0 column vectors to insert between consecutive data columns. TODO: Figure out if it would be better to do this w/ 1 as base or 0
        slice_stride_w (int): Number of 0 row vectors to insert between consecutive data columns. TODO: Same as above 
        conv_stride (int): Stride amount for convolution filter.
        input_dtype (DataType): Datatype for input. (Must be tkl.f16)
        output_dtype: (DataType): Datatype for output. (Must be tkl.f16)
        mem_space (tkl.IndexSymbol): Set GLOBAL_ADDRESS_SPACE or SHARED_ADDRESS_SPACE memory.
        block_m (Optional[int] | None): Workgroup tile size m dim (m x NF).
        block_n (Optional[int] | None): Workgroup tile size n dim (n x NF).
        block_k (Optional[int] | None): Workgroup tile size k dim (reduction axis).
        ratio_m (Optional[int] | None): Number of waves used along M dim.
        ratio_n (Optional[int] | None): Number of waves used along NF dim.

    Returns:
        (tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]): Wave kernel and Symbol table."""

    assert input_dtype == tkl.f16, f"Unsupported input dtype: {input_dtype}"
    assert output_dtype == tkl.f32, f"Unsupported input dtype: {output_dtype}"
    
    # Currently no padding is supported so can only be 0
    padding = 0

    # Symbols
    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Flip symbols
    H_FLIP, W_FLIP = sym.H_FLIP, sym.W_FLIP

    # 0 slice insert symbols
    # H_OUT_UPSAMP, W_OUT_UPSAMP = sym.H_OUT_SLICE, sym.W_OUT_SLICE
    STRIDE_H, STRIDE_W = tkl.sym.STRIDE_H, tkl.sym.STRIDE_W


    # This includes the new size of the input matrix after the 0 vector insert, weight (filter) stays same dim
    H_UP = H * STRIDE_H
    W_UP = W * STRIDE_W
    H_OUT_CONV = (H * slice_stride_h + 2 * padding - HF) // conv_stride + 1
    W_OUT_CONV = (W * slice_stride_w + 2 * padding - WF) // conv_stride + 1
    # H_OUT_CONV = (H - 1) * STRIDE_H + HF
    # W_OUT_CONV = (W - 1) * STRIDE_W + WF

    SZ_OUT = H_OUT_CONV * W_OUT_CONV

    # Shape for upsampling to be distrubited on block_m and block_n
    M0 = STRIDE_H * H * STRIDE_W * W * N

    # reduction dimension
    K = HF * WF * C
    # Shape for conv to be distrubited on block_m and block_n
    M = SZ_OUT * N
    

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)

    upsamp_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={N: i, C: j, H: k, W: l},
        outputs={
            N: i,
            C: j,
            H_UP: k * STRIDE_H,
            W_UP: l * STRIDE_W,
        },
    )
    conv_x_mapping  = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j % C,
            H_OUT_CONV: ((i % SZ_OUT) % W_OUT_CONV * conv_stride + (j // C) % WF), #* STRIDE_H,
            W_OUT_CONV: ((i % SZ_OUT) // W_OUT_CONV * conv_stride + (j // C) // WF), #* STRIDE_W,
        },
        outputs={M: i, K: j},
    )

    conv_w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, 
                C: j % C, 
                HF: H - 1 - ((j // C) // WF), # flip HF and WF and ordering
                WF: W - 1 - (j // C) % WF},
        outputs={NF: i, K: j},
    )

    conv_out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT_CONV: ((i % SZ_OUT) % W_OUT_CONV) * STRIDE_H,
            W_OUT_CONV: (i % SZ_OUT) // W_OUT_CONV * STRIDE_W,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K

    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    if layout == "nchw_fchw":
        x_type = tkl.Memory[N, C, H, W, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, NF, H_OUT_CONV, W_OUT_CONV, GLOBAL_ADDRESS_SPACE, output_dtype]
    elif layout == "nhwc_hwcf":
        x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, H_OUT_CONV, W_OUT_CONV, NF, GLOBAL_ADDRESS_SPACE, output_dtype]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    if block_m is None:
        block_m = 16

    if block_n is None:
        block_n = 16

    if block_k is None:
        block_k = 16

    if ratio_m is None:
        ratio_m = 1

    if ratio_n is None:
        ratio_n = 1


    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M0, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M0, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N / ratio_n)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_n, ratio_m, 1),
            vector_shapes={N: 16, H: 16, W: 16, C: 16} # H_OUT_UPSAMP: 1, W_OUT_UPSAMP: 1, H_FLIP: 1, W_FLIP: 1},
        )
    ]

    @tkw.wave(constraints)
    def trans_conv(
        x_raw: x_type,
        we_raw: we_type,
        slice_stride_h: tkl.i32,
        slice_stride_w: tkl.i32,
        final_out: out_type
    ) -> None:
        tkw.set_symbol(STRIDE_H, slice_stride_h)
        tkw.set_symbol(STRIDE_W, slice_stride_w)
        # need to use memory to store x
        # can intermediate results to val and use breakpoint 
        x_up_zeros_reg = tkl.Register[M0, N, input_dtype](0.0)
        # Allocate memory with 0's
        shape = (M0, N)
        shape = (N, C, H_UP, W_UP)
        x_up_zeros = allocate(shape, distributed_shape=(BLOCK_M, BLOCK_N), dtype=input_dtype, address_space=mem_space)
        # write 0 reg to memory
        tkw.write(x_up_zeros_reg, x_up_zeros)
        # read input matrix
        x_input = tkw.read(x_raw)
        # Use index mapping to map orginal matrix to upsampled matrix
        tkw.write(x_input, x_up_zeros, elements_per_thread=ELEMS_PER_THREAD, mapping=upsamp_mapping)

        c_reg = tkl.Register[M, NF, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, NF, output_dtype],
        ) -> tkl.Register[M, NF, output_dtype]:
            a_reg = tkw.read(
                x_up_zeros,
                mapping=conv_x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we_raw,
                mapping=conv_w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, final_out, mapping=conv_out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )
 
        
    symbols = {
        N: n,
        C: c,
        W: w,
        H: h,
        NF: nf,
        WF: wf,
        HF: hf,
        H_FLIP: h,
        W_FLIP: w,
        # H_OUT_UPSAMP: h * slice_stride_h,
        # W_OUT_UPSAMP: w * slice_stride_w,
        STRIDE_H: slice_stride_h,
        STRIDE_W: slice_stride_w,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: mem_space,
    }
    return trans_conv, symbols


from torch.testing import assert_close

def upsample_with_zeros(x, stride_h, stride_w):
    N, C, H, W = x.shape
    # H_out = (H - 1) * stride_h + 1
    # W_out = (W - 1) * stride_w + 1
    H_out = H * stride_h
    W_out = W * stride_w
    out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    # out[:, :, ::stride_h, ::stride_w] = x
    for ni in range(N):
        for hi in range(H):
            for wi in range(W):
                for ci in range(C):
                    out[ni, ci, hi * stride_h, wi * stride_w] = x[ni, ci, hi, wi]
    return out

import torch.nn.functional as F

if __name__ == "__main__":
    n, h, w, c = 1, 4, 5, 1
    nf, hf, wf, cf = 1, 3, 3, 1
    slice_stride_h, slice_stride_w = 1, 1
    padding = 0
    output_padding = 0

    # Input and filter
    x = device_randn(n, c, h, w, dtype=torch.float16)
    we = device_randn(nf, cf, hf, wf, dtype=torch.float16)

    # Reference manual transposed conv
    x_up = upsample_with_zeros(x, slice_stride_h, slice_stride_w)
    we_flipped = torch.flip(we, dims=[2, 3])
    # out_ref = torch.nn.Conv2d(x_up, we_flipped, padding=padding)
    convRef = torch.nn.Conv2d(c, nf, hf, stride=1, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we_flipped)
    out_ref = convRef(x_up).detach().to(torch.float32)

    # Print results
    print("Input (x):")
    print(x[0, 0])
    print("\nUpsampled Input:")
    print(x_up[0, 0])
    print("\nWeight:")
    print(we[0, 0])
    print("\nWeight (flipped):")
    print(we_flipped[0, 0])
 
    print("\nManual Transposed Convolution Output:")
    print(out_ref)
    print(out_ref.shape)
    layout = "nchw_fchw" 
    #layout = "nhwc_hwcf"

    if layout == "nchw_fchw":
        pass  # Nothing
    elif layout == "nhwc_hwcf":
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        we = torch.permute(we, (2, 3, 1, 0)).contiguous()
        out_ref = torch.permute(out_ref, (0, 2, 3, 1)).contiguous()
    else:
        raise ValueError(f"Invalid layout: {layout}")
    # Get compiled IREE kernel
    trans_conv, hyperparams = get_transponse_conv2d(
        layout=layout,
        n=n,
        h=h,
        w=w,
        c=c,
        hf=hf,
        wf=wf,
        nf=nf,
        slice_stride_h=slice_stride_h,
        slice_stride_w=slice_stride_w,
        conv_stride=1,
        input_dtype=tkl.f16,
        output_dtype=tkl.f32,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    trans_conv = wave_compile(options, trans_conv)

    out = torch.zeros_like(out_ref)
    trans_conv(x, we, slice_stride_h, slice_stride_w, out)
    # put breakpoint here after doing write in kernel to see what the value is
    print(f"\nWave Result:\n{out}")
    print(out.shape)

    assert_close(out, out_ref, rtol=1e-03, atol=1e-03)
