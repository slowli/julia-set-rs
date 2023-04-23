# Julia Set Computation and Rendering

[![Build Status](https://github.com/slowli/julia-set-rs/workflows/CI/badge.svg?branch=master)](https://github.com/slowli/julia-set-rs/actions)
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/julia-set-rs.svg)](https://github.com/slowli/julia-set-rs/blob/master/LICENSE)
![rust 1.65+ required](https://img.shields.io/badge/rust-1.65+-blue.svg?label=Required%20Rust)

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

<!-- FIXME: add code snippet -->

See the crate docs for the examples of usage.

### Installing Backend Dependencies

Note that OpenCL and Vulkan backends require the corresponding platform installed
in the execution environment. You may consult platform docs or [`ocl`] / [`vulkano`] crate
docs for possible installation options.

For quick testing, one may use [POCL](https://github.com/pocl/pocl);
it is an open source OpenCL implementation not tied to hardware 
(at the cost of being CPU-based, i.e., orders of magnitude
slower than OpenCL implementations by GPU vendors).
POCL [can be installed from sources](http://portablecl.org/docs/html/install.html)
with the commands like in the [installation script](install-pocl.sh)
(tested on Ubuntu 22.04).

## License

Licensed under the [Apache-2.0 license](LICENSE).

[Julia set]: https://en.wikipedia.org/wiki/Julia_set
[OpenCL]: https://www.khronos.org/opencl/
[Vulkan]: https://www.khronos.org/vulkan/
[`ocl`]: https://crates.io/crates/ocl
[`vulkano`]: https://crates.io/crates/vulkano
