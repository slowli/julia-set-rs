# Julia Set Computation and Rendering

[![License: Apache-2.0](https://img.shields.io/github/license/slowli/julia-set-rs.svg)](https://github.com/slowli/julia-set-rs/blob/master/LICENSE)

**Documentation:**
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/julia-set-rs/julia_set/)

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
