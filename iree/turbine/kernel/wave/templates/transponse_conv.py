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
import torch.nn.functional as F
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

def get_transponse_conv2d(
    layout: str,
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    upsamp_stride_h: int,
    upsamp_stride_w: int,
    conv_stride: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    ratio_m: Optional[int] = None,
    ratio_n: Optional[int] = None,
) -> tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]:
    assert input_dtype == tkl.f16, f"Unsupported input dtype: {input_dtype}"
    assert output_dtype == tkl.f32, f"Unsupported input dtype: {output_dtype}"
    padding = 0  # TODO: only pad=0 is supported for now

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_UP = H * upsamp_stride_h
    W_UP = W * upsamp_stride_w

    STRIDE_H, STRIDE_W = tkl.sym.STRIDE_H, tkl.sym.STRIDE_W

    H_OUT = (H * upsamp_stride_h + 2 * padding - HF) // conv_stride + 1
    W_OUT = (W * upsamp_stride_w + 2 * padding - WF) // conv_stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N
    # Shape for upsampling to be distrubited on block_m and block_n
    M0 = STRIDE_H * H * STRIDE_W * W * N

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

    # Align C dim reading pattern to be contiguous for nhwc_hwcf pattern.
    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j % C,
            H_UP: (i % SZ_OUT) % W_OUT * conv_stride + (j // C) % WF,
            W_UP: (i % SZ_OUT) // W_OUT * conv_stride + (j // C) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j % C, HF: HF - 1 - (j // C) % WF, WF: WF - 1 - (j // C) // WF},
        outputs={NF: i, K: j},
    )
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    if layout == "nchw_fchw":
        x_type = tkl.Memory[N, C, H, W, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, NF, H_OUT, W_OUT, GLOBAL_ADDRESS_SPACE, output_dtype]
    elif layout == "nhwc_hwcf":
        x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, output_dtype]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if block_m is None:
        block_m = 64

    if block_n is None:
        block_n = 128

    if block_k is None:
        block_k = 32

    if ratio_m is None:
        ratio_m = 2

    if ratio_n is None:
        ratio_n = 2

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M0, BLOCK_M, 1, primary=False)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N / ratio_n)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_n, ratio_m, 1),
            vector_shapes={N: 1, H: 1, W: 1, C: 1}

        )
    ]

    @tkw.wave(constraints)
    def conv(
        x: x_type,
        we: we_type,
        upsamp_stride_h: tkl.i32,
        upsamp_stride_w: tkl.i32,
        out: out_type,
    ):
        tkw.set_symbol(STRIDE_H, upsamp_stride_h)
        tkw.set_symbol(STRIDE_W, upsamp_stride_w)
        shape = (N, C, H_UP, W_UP)
        #shape = (M0, N)
        x_up_zeros_reg = tkl.Register[H, C, H_UP, W_UP, input_dtype](0.0)

        x_up_zeros = allocate(shape, distributed_shape=(H, C, H_UP, W_UP), dtype=input_dtype, address_space=mem_space)
        
        tkw.write(x_up_zeros_reg, x_up_zeros)
        x_input = tkw.read(x)
        tkw.write(x_input, x_up_zeros, elements_per_thread=ELEMS_PER_THREAD, mapping=upsamp_mapping)

        c_reg = tkl.Register[M, NF, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, NF, output_dtype],
        ) -> tkl.Register[M, NF, output_dtype]:
            a_reg = tkw.read(
                x_up_zeros,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    symbols = {
        N: n,
        C: c,
        W: w,
        H: h,
        NF: nf,
        WF: wf,
        HF: hf,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: mem_space,
    }

    return conv, symbols


from torch.testing import assert_close

def upsample_with_zeros(x, stride_h, stride_w):
    N, C, H, W = x.shape
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


if __name__ == "__main__":
    n, h, w, c = 1, 4, 4, 1
    nf, hf, wf, cf = 1, 2, 2, 1
    upsamp_stride_h, upsamp_stride_w = 1, 1
    padding = 0
    output_padding = 0

    # Input and filter
    x = device_randn(n, c, h, w, dtype=torch.float16)
    we = device_randn(nf, cf, hf, wf, dtype=torch.float16)
    we_flipped = torch.flip(we, dims=[2, 3])
    # Reference manual transposed conv
    x_up = upsample_with_zeros(x, upsamp_stride_h, upsamp_stride_w)
    # out_ref = torch.nn.Conv2d(x_up, we_flipped, padding=padding)
    convRef = torch.nn.Conv2d(c, nf, hf, stride=1, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we_flipped)
    out_ref = convRef(x_up).detach().to(torch.float32)


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
        upsamp_stride_h=upsamp_stride_h,
        upsamp_stride_w=upsamp_stride_w,
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
    trans_conv(x, we, upsamp_stride_h, upsamp_stride_w, out)

    # Print results
    print("Input (x):")
    print(x[0, 0])
    print("\nUpsampled Input:")
    print(x_up[0, 0])
    print("\nWeight:")
    print(we[0, 0])
 
    print("\nManual Transposed Convolution Output:")
    print(out_ref)
    print(out_ref.shape)
    print(f"\nWave Result:\n{out}")
    print(out.shape)

    assert_close(out, out_ref, rtol=1e-03, atol=1e-03)