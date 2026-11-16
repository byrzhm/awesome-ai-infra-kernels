# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, etc.) working in this repository.

## What this repository is

`awesome-ai-infra-kernels` is a thin meta-repository that integrates multiple high-performance AI infrastructure kernel libraries as git submodules under `kernels/`. The root contains only documentation, `LICENSE` and `.gitmodules` — there is no first-party source code or build system at the root.

Each library under `kernels/` is an independent repository with its own history, license, build system and upstream.

## Repository layout

```
├── README.md          # repository overview and library index
├── AGENTS.md          # this file
├── LICENSE            # MIT — covers this repo only, not submodules
├── .gitmodules        # submodule registry
└── kernels/
    ├── DeepGEMM/           # submodule → byrzhm/DeepGEMM (fork of deepseek-ai/DeepGEMM)
    ├── cutlass/            # submodule → byrzhm/cutlass (fork of NVIDIA/cutlass)
    ├── DeepEP/             # submodule → byrzhm/DeepEP (fork of deepseek-ai/DeepEP)
    ├── MoonEP/             # submodule → byrzhm/MoonEP (fork of MoonshotAI/MoonEP)
    ├── flash-attention/    # submodule → byrzhm/flash-attention (fork of dao-AILAB/flash-attention)
    └── TransformerEngine/  # submodule → byrzhm/TransformerEngine (fork of NVIDIA/TransformerEngine)
```

## Working with submodules

- After cloning or pulling the root repo, always run `git submodule update --init --recursive` before doing anything else.
- Some libraries have their own nested submodules (DeepGEMM and DeepEP vendor DeepJIT; DeepGEMM, flash-attention and TransformerEngine vendor CUTLASS; flash-attention also vendors ROCm's `composable_kernel` and `aiter`; TransformerEngine vendors googletest and nccl-extensions); the recursive update above handles them too.
- The root repo records only a gitlink (pinned commit) per library. To update a library: check out the target commit inside `kernels/<Lib>`, then `git add kernels/<Lib>` from the root and commit.
- Files inside `kernels/<Lib>` belong to that library's own repository. Do not modify them as part of a root-repo change unless the user explicitly asks; such changes belong in the library's own repository (the fork), not here.
- Never commit build artifacts or generated files.

## Build and run constraints

- The integrated libraries are CUDA kernel libraries targeting Linux + NVIDIA SM90/SM100 GPUs (CUDA Toolkit 12.9+). They cannot be built or run on macOS.
- Do not run builds, tests or benchmarks inside a submodule on a machine without a suitable GPU. If GPU work is needed, it happens on the user's Linux GPU machines — ask the user where to run it.

## Hard rules

- **禁止 Agent push 到华为云 CodeHub。** Never push this repository or any of its submodules to Huawei Cloud CodeHub, and never add a CodeHub remote.
- Never `git push` on your own initiative. The only allowed remote is the GitHub origin (`git@github.com:byrzhm/awesome-ai-infra-kernels.git`), and only when the user explicitly asks.
- Do not commit or push inside a submodule unless the user explicitly asks.

## Conventions

- New libraries are added with `git submodule add <url> kernels/<Name>`, and must be accompanied by an entry in the "Included libraries" table in `README.md` plus a short section describing the library.
- Commit the `.gitmodules` change and the new `kernels/<Name>` gitlink together in a single commit.
- Submodule URLs use the SSH form (`git@github.com:...`).
- Keep the root minimal: documentation only, no scripts or build files unless the user asks for them.
