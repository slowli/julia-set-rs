# Julia Set Computation and Rendering

[![Build Status](https://github.com/slowli/julia-set-rs/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/julia-set-rs/actions)
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/julia-set-rs.svg)](https://github.com/slowli/julia-set-rs/blob/master/LICENSE)
![rust 1.44.0+ required](https://img.shields.io/badge/rust-1.44.0+-blue.svg?label=Required%20Rust)

**Documentation:**
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/julia-set-rs/julia_set/)

<p>
  <img src="https://github.com/slowli/julia-set-rs/raw/master/examples/tiles.png" alt="Fractal example" width="480" height="240">
</p>

Library to compute and render the [Julia set] boundary for complex-valued functions
and render it to an image. Depending on the function, rendered images frequently
have a fractal-like nature.

## Features

- Supports multi-threaded CPU, [OpenCL] and [Vulkan] backends.
- Allows using custom complex-valued functions (not only *boring* quadratic ones).
- Supports customizable rendering params (e.g., the rendered area).
- Allows reusing the same compiled program with different rendering params,
  thus saving time on OpenCL / Vulkan shader compilation.
- Comes with a [CLI example](examples/cli.rs) for a quickstart.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
julia-set = "0.1.0"
```

See the crate docs for the examples of usage.

### Installing Backend Dependencies

Note that OpenCL and Vulkan backends require the corresponding platform installed
in the execution environment. You may consult platform docs or [`ocl`] / [`vulkano`] crate
docs for possible installation options.

For quick testing, one may use [POCL](https://github.com/pocl/pocl);
it is an open source OpenCL implementation not tied to hardware 
(at the cost of being CPU-based, i.e., orders of magnitude
slower than OpenCL implementations by GPU vendors).
POCL may be installed from sources with the commands like these
(showcased here for Ubuntu Bionic):

```bash
# Install utils for build
apt-get install build-essential cmake pkg-config libhwloc-dev zlib1g-dev
# Install OpenCL-related utils
apt-get install ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev clinfo
# Install LLVM / Clang from the official APT repository
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - 
add-apt-repository 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
apt-get update
apt-get install clang-9 libclang-9-dev llvm-9 llvm-9-dev

# Get POCL sources
export POCL_VER=1.5 # latest stable version
curl -sSL "https://github.com/pocl/pocl/archive/v$POCL_VER.tar.gz" > pocl-$POCL_VER.tar.gz
tar xf "pocl-$POCL_VER.tar.gz"
# Build POCL from the sources
cd pocl-$POCL_VER
mkdir build && cd build
cmake -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-9 -DCMAKE_INSTALL_PREFIX=/usr ..
make

# Verify installation
clinfo
# If successful, `clinfo` should display information about the POCL platform.
```

## License

Licensed under the [Apache-2.0 license](LICENSE).

[Julia set]: https://en.wikipedia.org/wiki/Julia_set
[OpenCL]: https://www.khronos.org/opencl/
[Vulkan]: https://www.khronos.org/vulkan/
[`ocl`]: https://crates.io/crates/ocl
[`vulkano`]: https://crates.io/crates/vulkano
