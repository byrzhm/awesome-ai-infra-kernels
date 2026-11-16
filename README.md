# awesome-ai-infra-kernels

High-performance AI infrastructure kernel libraries, integrated into a single repository as git submodules — one clone, pinned versions, easy management, benchmarking, and side-by-side study of kernel implementations.

This is a thin "meta" repository: all kernel code lives in submodules under [`kernels/`](kernels/), each pinned to a specific commit and keeping its own license and build system. The root only holds documentation and the submodule registry.

## Included libraries

| Library | Path | Upstream | License | Tracked branch |
|---|---|---|---|---|
| [DeepGEMM](kernels/DeepGEMM) | `kernels/DeepGEMM` | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | MIT | `main` |
| [CUTLASS](kernels/cutlass) | `kernels/cutlass` | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | BSD-3-Clause | `main` |
| [DeepEP](kernels/DeepEP) | `kernels/DeepEP` | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | MIT | `main` |
| [MoonEP](kernels/MoonEP) | `kernels/MoonEP` | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) | MIT | `master` |
| [FlashAttention](kernels/flash-attention) | `kernels/flash-attention` | [dao-AILAB/flash-attention](https://github.com/dao-AILAB/flash-attention) | BSD-3-Clause | `main` |
| [TransformerEngine](kernels/TransformerEngine) | `kernels/TransformerEngine` | [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | Apache-2.0 | `main` |

### DeepGEMM

DeepSeek's unified, high-performance tensor core kernel library, bringing together the key computation primitives of modern LLMs into a single CUDA codebase:

- **GEMMs** — FP8, FP4 and BF16; dense, contiguous-grouped and masked-grouped (MoE) layouts, plus K-axis-grouped for MoE weight backward
- **Mega MoE** — fused expert-parallel dispatch/combine, two linear layers (FP8xFP4 / FP8xFP8), SwiGLU and NVLink communication overlap in a single mega-kernel
- **Indexer kernels** — weighted ReLU MQA logits scoring (non-paged and paged) for the lightning indexer
- **DeepJIT** — all kernels are compiled at runtime, no CUDA compilation during installation

It leverages concepts from CUTLASS/CuTe but avoids heavy reliance on their templates, keeping a small set of core kernel functions — a clean resource for learning NVIDIA GPU kernel optimization.

Requirements: NVIDIA SM90 (Hopper) or SM100 (Blackwell) GPU, CUDA Toolkit 12.9+, PyTorch 2.3+.

The submodule points to the [byrzhm/DeepGEMM](https://github.com/byrzhm/DeepGEMM) fork, which tracks upstream `main` (currently at the same commit, `5081f85`). See [kernels/DeepGEMM/README.md](kernels/DeepGEMM/README.md) for full details.

### CUTLASS

NVIDIA's CUDA templates for high-performance GEMM and related computations — the reference implementation of hierarchical kernel decomposition (CTA → warp → instruction level), data movement (TMA, async copy) and epilogue fusion, built from reusable, modular software components.

- **C++ templates** — since 2017, with extensive data-type coverage: FP64/FP32/TF32/FP16/BF16, FP8 (e4m3/e5m2), block-scaled NVFP4 and MXFP4/MXFP6/MXFP8, narrow integers and 1-bit binary types, across Volta, Turing, Ampere, Ada, Hopper and Blackwell
- **CUTLASS DSLs (CuTe DSL)** — Python-native kernel authoring on the same core CUTLASS/CuTe concepts: much faster compile times, no C++ glue code, and metaprogramming without deep C++ expertise
- Pinned at 4.8.1 (Oct. 2026)

The submodule points to the [byrzhm/cutlass](https://github.com/byrzhm/cutlass) fork, which tracks upstream `main` (currently at the same commit, `b2f249ac`). See [kernels/cutlass/README.md](kernels/cutlass/README.md) for full details.

> **Note:** DeepGEMM, FlashAttention and TransformerEngine all vendor CUTLASS through their own nested submodules (`kernels/DeepGEMM/third-party/cutlass`, `kernels/flash-attention/csrc/cutlass`, `kernels/TransformerEngine/3rdparty/cutlass`), so a fully recursive clone fetches multiple independent copies of CUTLASS.

### DeepEP

DeepSeek's high-performance communication library for machine learning training and inference:

- **Expert parallelism** — high-throughput and low-latency all-to-all GPU kernels for MoE dispatch and combine (FP8 dispatch supported), unified under the `EPBuffer` interface since V2
- **Redundant-expert exchange** — NVLink weight prefetch (`lb_prefetch_weights`) and gradient reduction (`lb_reduce_grads`) for dynamic redundant experts
- **Experimental primitives** — pipeline parallelism (PP), context/data parallelism bucket collectives (CP/DP), and remote memory access (Engram)
- Communication kernels are JIT-compiled at runtime via DeepJIT; NCCL Gin backend (NVSHMEM removed as a dependency in V2.5)

The submodule points to the [byrzhm/DeepEP](https://github.com/byrzhm/DeepEP) fork, which tracks upstream `main` (currently at the same commit, `f6aa38f`). See [kernels/DeepEP/README.md](kernels/DeepEP/README.md) for full details.

### MoonEP

Moonshot AI's expert-parallelism communication library that keeps token loads perfectly balanced across ranks via dynamic redundant experts:

- **Perfect balance** — every rank receives exactly `S × K` tokens no matter how skewed the routing is: a small set of redundant experts is planned online from the current router outputs and prefetched before expert computation, with gradients reduced back to home ranks in backward
- **Zero copy and static shapes** — fused permute/unpermute: tokens are sent directly to their expert-grouped positions on remote ranks, buffer views are handed to the computation, and the fixed `S × K` buffer eliminates per-layer MoE host synchronization
- **Online planning** — a near-optimal GPU planning kernel with negligible overhead

Benchmarks on H20 (EP=8) show flat communication time under router imbalance where DeepEP v2's latency — set by the hottest rank — degrades, and no OOM from activation-shape fragmentation during end-to-end training. The redundant-expert approach pairs with DeepEP's `lb_prefetch_weights` / `lb_reduce_grads` primitives.

The submodule points to the [byrzhm/MoonEP](https://github.com/byrzhm/MoonEP) fork, which tracks upstream `master` (currently at the same commit, `d86fda4`). See [kernels/MoonEP/README.md](kernels/MoonEP/README.md) for full details.

### FlashAttention

The official implementation of FlashAttention — fast and memory-efficient exact attention with IO-awareness (Tri Dao et al.):

- **FlashAttention-4** (active development, `flash_attn/cute/`) — written in Python with CuTe DSL and JIT-compiled to PTX/CUBIN at runtime; targets Hopper (SM90) and Blackwell (SM100/SM110); varlen, causal and sliding-window masks, softcap, user-defined `score_mod`/`mask_mod`, block sparsity, paged KV, GQA/MQA, split-KV with combine
- **Earlier generations kept in-tree** — FA2 (CUDA C++, `csrc/`) and FA3 (Hopper-optimized, `hopper/`)

The submodule points to the [byrzhm/flash-attention](https://github.com/byrzhm/flash-attention) fork, which tracks upstream `main` (currently at the same commit, `2dd8952`). It has its own nested submodules — CUTLASS under `csrc/cutlass` plus ROCm's `composable_kernel` and `aiter` for the HIP backend. See [kernels/flash-attention/README.md](kernels/flash-attention/README.md) for full details.

### TransformerEngine

NVIDIA's library for accelerating Transformer model training on NVIDIA GPUs — optimized building blocks and fused kernels combined with automatic mixed-precision-style APIs for PyTorch and JAX, so low-precision training can be adopted without rewriting a training stack. TE manages the scaling factors, amax histories and quantization metadata required by low-precision recipes.

- **Low-precision training** — FP8 on Hopper, Ada, Blackwell and Rubin; MXFP8 and NVFP4 on Blackwell; FP16/BF16 optimizations from Ampere onward
- **Fused modules** — attention, linear layers, normalization, quantization, and fused Transformer and MoE building blocks
- **Distributed** — tensor, sequence, context and expert parallelism, with communication overlap
- **Architecture** — PyTorch and JAX frontends over a framework-independent common layer (CUDA kernels, public C API, dispatch)

The submodule points to the [byrzhm/TransformerEngine](https://github.com/byrzhm/TransformerEngine) fork, which tracks upstream `main` (currently at the same commit, `4f5c08e`). Nested submodules: CUTLASS, googletest and nccl-extensions under `3rdparty/`. See [kernels/TransformerEngine/README.rst](kernels/TransformerEngine/README.rst) for full details.

## Usage

### Clone

```bash
git clone --recursive git@github.com:byrzhm/awesome-ai-infra-kernels.git

# or, for an existing clone:
git submodule update --init --recursive
```

If you don't have SSH keys for GitHub, rewrite the submodule URLs on the fly:

```bash
git -c url."https://github.com/".insteadOf="git@github.com:" \
  clone --recursive https://github.com/byrzhm/awesome-ai-infra-kernels.git
```

### Update a library

```bash
cd kernels/DeepGEMM
git fetch origin && git checkout <commit-or-tag>   # or: git checkout origin/main
cd ../..
git add kernels/DeepGEMM
git commit -m "Update DeepGEMM to <commit-or-tag>"
```

Check the current state of all libraries with `git submodule status`.

### Build and run

Each library is built and tested with its own tooling inside its directory — see each library's README. These are CUDA libraries: they require Linux with an NVIDIA GPU (SM90/SM100) and a matching CUDA toolkit. They cannot be built on macOS.

## Adding a new library

```bash
git submodule add git@github.com:<owner>/<repo>.git kernels/<Name>
```

Then add it to the "Included libraries" table above (with a short section describing it), and commit the `.gitmodules` change together with the new `kernels/<Name>` gitlink in a single commit.

## License

The repository itself (documentation and submodule pointers) is released under the [MIT License](LICENSE). Each integrated library under `kernels/` retains its own license — check the submodule directory.
