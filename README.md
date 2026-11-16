# awesome-ai-infra-kernels

High-performance AI infrastructure kernel libraries, integrated into a single repository as git submodules — one clone, pinned versions, easy management, benchmarking, and side-by-side study of kernel implementations.

This is a thin "meta" repository: all kernel code lives in submodules under [`kernels/`](kernels/), each pinned to a specific commit and keeping its own license and build system. The root only holds documentation and the submodule registry.

## Included libraries

| Library | Path | Upstream | License | Tracked branch |
|---|---|---|---|---|
| [DeepGEMM](kernels/DeepGEMM) | `kernels/DeepGEMM` | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | MIT | `main` |

### DeepGEMM

DeepSeek's unified, high-performance tensor core kernel library, bringing together the key computation primitives of modern LLMs into a single CUDA codebase:

- **GEMMs** — FP8, FP4 and BF16; dense, contiguous-grouped and masked-grouped (MoE) layouts, plus K-axis-grouped for MoE weight backward
- **Mega MoE** — fused expert-parallel dispatch/combine, two linear layers (FP8xFP4 / FP8xFP8), SwiGLU and NVLink communication overlap in a single mega-kernel
- **Indexer kernels** — weighted ReLU MQA logits scoring (non-paged and paged) for the lightning indexer
- **DeepJIT** — all kernels are compiled at runtime, no CUDA compilation during installation

It leverages concepts from CUTLASS/CuTe but avoids heavy reliance on their templates, keeping a small set of core kernel functions — a clean resource for learning NVIDIA GPU kernel optimization.

Requirements: NVIDIA SM90 (Hopper) or SM100 (Blackwell) GPU, CUDA Toolkit 12.9+, PyTorch 2.3+.

The submodule points to the [byrzhm/DeepGEMM](https://github.com/byrzhm/DeepGEMM) fork, which tracks upstream `main` (currently at the same commit, `5081f85`). See [kernels/DeepGEMM/README.md](kernels/DeepGEMM/README.md) for full details.

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
