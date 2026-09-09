# Portable Source Package Notes

This directory is a cleaned copy of the C model source, split by purpose.

Top-level layout:

- `necessary_source/`: only the files needed to build the main program, suitable for upload as the minimal source package
- `experimental_trace_source/`: trace/replay experiment programs plus their required headers
- `docs/`: README/debug/precision notes

In this repository the C model folder is named `cmodel/`; the minimal upload/build folder is `cmodel/necessary_source/`. It keeps:

- CUDA/C++ source and header files: `*.cu`, `*.h`
- Build files: `Makefile`
- `.gitignore`

The docs folder keeps:

- `README.md`, `DEBUG_USAGE.md`, `precision.md`, `dtype_summary/**/*.md`

It intentionally excludes:

- Git metadata: `.git/`
- Compiled binaries: `all_fmac_test`, `trace_mismatch_log`, and similar executables
- Helper scripts: `*.sh`, `tools/*.py`
- Runtime output and trace data: `output_data*/`, `scale_output_data*/`, `debug_data*/`, `trace_*/`
- Log files and Python bytecode: `*.log`, `__pycache__/`

## Build

Software-model build, no NVIDIA MMA path:

```bash
make build NVIDIA_ON=0
```

Full NVIDIA MMA build:

```bash
make build NVIDIA_ON=1 CUDA_HOME=/usr/local/cuda CUTLASS_DIR=/path/to/cutlass CUDA_ARCH=120a
```

Useful variants:

```bash
make build_static_runtime NVIDIA_ON=1 CUDA_HOME=/usr/local/cuda CUTLASS_DIR=/path/to/cutlass CUDA_ARCH=120a
make build_256b NVIDIA_ON=1 CUDA_HOME=/usr/local/cuda CUTLASS_DIR=/path/to/cutlass CUDA_ARCH=120a
```

The trace experiment programs are intentionally separated under `experimental_trace_source/`.

`CUDA_ARCH` defaults to `120a`, matching the original Blackwell-oriented build. Change it if the target GPU requires a different architecture.

## Dependencies

- Linux shell environment
- `g++` for `NVIDIA_ON=0`
- CUDA toolkit and CUTLASS for `NVIDIA_ON=1`

The Makefile defaults are portable:

- `CUDA_HOME ?= /usr/local/cuda`
- `CUTLASS_DIR ?= $(HOME)/cutlass`
- `CUDA_ARCH ?= 120a`

Override these variables on the `make` command line when building elsewhere.

## Run

After build:

```bash
./all_fmac_test --help
./all_fmac_test --list-dtypes
./all_fmac_test --dtype 1 --vec-count 1000
```

Long-regression launcher scripts are intentionally not included in this source-only package. Add your own scripts around the `make` targets and `all_fmac_test` command-line options when needed.
