# VS Code Setup

The repository includes two C/C++ IntelliSense configurations in
`.vscode/c_cpp_properties.json`. VS Code loads them when the repository root
is opened, but each developer selects the configuration appropriate for the
current host once through the C/C++ configuration selector in the status bar.

## macOS

1. Install the recommended `C/C++` extension when VS Code prompts for it.
2. Open the repository root, not an individual source file.
3. Select `macOS C-model` from the C/C++ configuration selector.
4. Run `C/C++: Reset IntelliSense Database`, then reload the VS Code window.

This defines `NVIDIA_ON=0`, matching the portable C-model build:

```sh
make -C cmodel/necessary_source build NVIDIA_ON=0
```

CUDA MMA headers are deliberately excluded from the active include chain on
macOS.

## Windows CUDA

1. Install the recommended `C/C++` extension.
2. Install the CUDA toolkit and CUTLASS, then set `CUDA_PATH` and `CUTLASS_DIR`
   in the environment visible to VS Code.
3. Open the repository root and select `Windows CUDA` from the C/C++
   configuration selector.
4. Run `C/C++: Reset IntelliSense Database`, then reload the VS Code window.

This defines `NVIDIA_ON=1` and adds the CUDA/CUTLASS include directories. Build
the CUDA path with the existing Makefile command in a CUDA-capable environment:

```sh
make -C cmodel/necessary_source build NVIDIA_ON=1 CUDA_HOME=... CUTLASS_DIR=...
```

The shared configuration does not install CUDA, CUTLASS, or a Make toolchain.
