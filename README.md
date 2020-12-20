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

## License

Licensed under the [Apache-2.0 license](LICENSE).

[Julia set]: https://en.wikipedia.org/wiki/Julia_set
[OpenCL]: https://www.khronos.org/opencl/
[Vulkan]: https://www.khronos.org/vulkan/
