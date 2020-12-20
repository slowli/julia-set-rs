//! [Julia set] boundary computation and rendering.
//!
//! # Theory
//!
//! Informally, the Julia set for a complex-valued function `f` (in Rust terms,
//! `fn(Complex32) -> Complex32`) is a set of complex points for which an infinitely small
//! perturbation can lead to drastic changes in the sequence of iterated function applications
//! (that is, `f(z)`, `f(f(z))`, `f(f(f(z)))` and so on).
//!
//! For many functions `f`, the iterated sequence may tend to infinity. Hence, the
//! commonly used computational way to render the Julia set boundary is as follows:
//!
//! 1. For each complex value `z` within a rectangular area, perform steps 2-3.
//! 2. Compute the minimum iteration `0 < i <= MAX_I` such that `|f(f(f(...(z)))| > R`.
//!   Here, `f` is applied `i` times; `R` is a positive real-valued constant
//!   (the *infinity distance*); `MAX_I` is a positive integer constant (maximum iteration count).
//! 3. Associate `z` with a color depending on `i`. For example, `i == 1` may be rendered as black,
//!   `i == MAX_I` as white, and values between it may get the corresponding shades of gray.
//! 4. Render the rectangular area as a (two-dimensional) image, with each pixel corresponding
//!   to a separate value of `z`.
//!
//! This is exactly the way Julia set rendering is implemented in this crate.
//!
//! [Julia set]: https://en.wikipedia.org/wiki/Julia_set
//!
//! # Backends
//!
//! The crate supports several computational [`Backend`]s.
//!
//! | Backend | Crate feature | Hardware | Crate dep(s) |
//! |---------|---------------|----------|------------|
//! | [`OpenCl`] | `opencl_backend` | GPU, CPU | [`ocl`] |
//! | [`Vulkan`] | `vulkan_backend` | GPU | [`vulkano`], [`shaderc`] |
//! | [`Cpu`] | `cpu_backend` | CPU | [`rayon`] |
//! | [`Cpu`] | `dyn_cpu_backend` | CPU | [`rayon`] |
//!
//! None of the backends are on by default. A backend can be enabled by switching
//! on the corresponding crate feature. `dyn_cpu_backend` requires `cpu_backend` internally.
//!
//! All backends except for `cpu_backend` require parsing the complex-valued [`Function`] from
//! a string presentation, e.g., `"z * z - 0.4i"`. The [`arithmetic-parser`] crate is used for this
//! purpose. For `cpu_backend`, the function is defined directly in Rust.
//!
//! For efficiency and modularity, a [`Backend`] creates a *program* for each function.
//! (In case of OpenCL, a program is a kernel, and in Vulkan a program is a compute shader.)
//! The program can then be [`Render`]ed with various [`Params`].
//!
//! Backends targeting GPUs (i.e., `OpenCl` and `Vulkan`) should be much faster than CPU-based
//! backends. Indeed, the rendering task is [embarrassingly parallel] (could be performed
//! independently for each point).
//!
//! [`ocl`]: https://crates.io/crates/ocl
//! [`vulkano`]: https://crates.io/crates/vulkano
//! [`shaderc`]: https://crates.io/crates/shaderc
//! [`rayon`]: https://crates.io/crates/rayon
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [embarrassingly parallel]: https://en.wikipedia.org/wiki/Embarrassingly_parallel
//!
//! # Examples
//!
//! Using Rust function definition with `cpu_backend`:
//!
//! ```
//! use julia_set::{Backend, Cpu, Params, Render};
//! use num_complex::Complex32;
//!
//! # fn main() -> anyhow::Result<()> {
//! let program = Cpu.create_program(|z: Complex32| z * z + Complex32::new(-0.4, 0.5))?;
//! let render_params = Params::new([50, 50], 4.0).with_infinity_distance(5.0);
//! let image = program.render(&render_params)?;
//! // Do something with the image...
//! # Ok(())
//! # }
//! ```
//!
//! Using interpreted function definition with `dyn_cpu_backend`:
//!
//! ```
//! use julia_set::{Backend, Cpu, Function, Params, Render};
//! use num_complex::Complex32;
//!
//! # fn main() -> anyhow::Result<()> {
//! let function: Function = "z * z - 0.4 + 0.5i".parse()?;
//! let program = Cpu.create_program(&function)?;
//! let render_params = Params::new([50, 50], 4.0).with_infinity_distance(5.0);
//! let image = program.render(&render_params)?;
//! // Do something with the image...
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/julia-set/0.1.0")]
#![warn(missing_docs, missing_debug_implementations, bare_trait_objects)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::doc_markdown
)]

use std::fmt;

#[cfg(feature = "cpu_backend")]
pub use crate::cpu::{ComputePoint, Cpu, CpuProgram};
#[cfg(feature = "arithmetic-parser")]
pub use crate::function::{FnError, Function};
#[cfg(feature = "opencl_backend")]
pub use crate::opencl::{OpenCl, OpenClProgram};
#[cfg(feature = "vulkan_backend")]
pub use crate::vulkan::{Vulkan, VulkanProgram};

#[cfg(any(feature = "opencl_backend", feature = "vulkan_backend"))]
mod compiler;
#[cfg(feature = "cpu_backend")]
mod cpu;
#[cfg(feature = "arithmetic-parser")]
mod function;
#[cfg(feature = "opencl_backend")]
mod opencl;
pub mod transform;
#[cfg(feature = "vulkan_backend")]
mod vulkan;

/// Image buffer output by a [`Backend`].
pub type ImageBuffer = image::GrayImage;

/// Backend capable of converting an input (the type parameter) into a program. The program
/// then can be used to [`Render`] the Julia set with various rendering [`Params`].
pub trait Backend<In>: Default {
    /// Error that may be returned during program creation.
    type Error: fmt::Debug + fmt::Display;
    /// Program output by the backend.
    type Program: Render;

    /// Creates a program.
    ///
    /// # Errors
    ///
    /// May return an error if program cannot be created (out of memory, etc.).
    fn create_program(&self, function: In) -> Result<Self::Program, Self::Error>;
}

/// Program for a specific [`Backend`] (e.g., OpenCL) corresponding to a specific Julia set.
/// A single program can be rendered with different parameters (e.g., different output sizes),
/// but the core settings (e.g., the complex-valued function describing the set) are fixed.
pub trait Render {
    /// Error that may be returned during rendering.
    type Error: fmt::Debug + fmt::Display;

    /// Renders the Julia set with the specified parameters.
    ///
    /// # Errors
    ///
    /// May return an error if the backend does not support rendering with the specified params
    /// or due to external reasons (out of memory, etc.).
    fn render(&self, params: &Params) -> Result<ImageBuffer, Self::Error>;
}

/// Julia set rendering parameters.
///
/// The parameters are:
///
/// - Image dimensions (in pixels)
/// - View dimensions and view center determining the rendered area. (Only the view height
///   is specified explicitly; the width is inferred from the height and
///   the image dimension ratio.)
/// - Infinity distance
/// - Upper bound on the iteration count
///
/// See the [Julia set theory] for more details regarding these parameters.
///
/// [Julia set theory]: index.html#theory
#[derive(Debug, Clone)]
pub struct Params {
    view_center: [f32; 2],
    view_height: f32,
    inf_distance: f32,
    image_size: [u32; 2],
    max_iterations: u8,
}

impl Params {
    /// Creates a new set of params with the specified `image_dimensions` and the `view_height`
    /// of the rendered area.
    ///
    /// The remaining parameters are set as follows:
    ///
    /// - The width of the rendered area is inferred from these params.
    /// - The view is centered at `0`.
    /// - The infinity distance is set at `3`.
    ///
    /// # Panics
    ///
    /// Panics if any of the following conditions do not hold:
    ///
    /// - `image_dimensions` are positive
    /// - `view_height` is positive
    pub fn new(image_dimensions: [u32; 2], view_height: f32) -> Self {
        assert!(image_dimensions[0] > 0);
        assert!(image_dimensions[1] > 0);
        assert!(view_height > 0.0, "`view_height` should be positive");

        Self {
            view_center: [0.0, 0.0],
            view_height,
            inf_distance: 3.0,
            image_size: image_dimensions,
            max_iterations: 100,
        }
    }

    /// Sets the view center.
    pub fn with_view_center(mut self, view_center: [f32; 2]) -> Self {
        self.view_center = view_center;
        self
    }

    /// Sets the infinity distance.
    ///
    /// # Panics
    ///
    /// Panics if the provided distance is not positive.
    pub fn with_infinity_distance(mut self, inf_distance: f32) -> Self {
        assert!(inf_distance > 0.0, "`inf_distance` should be positive");
        self.inf_distance = inf_distance;
        self
    }

    /// Sets the maximum iteration count.
    ///
    /// # Panics
    ///
    /// Panics if `max_iterations` is zero.
    pub fn with_max_iterations(mut self, max_iterations: u8) -> Self {
        assert_ne!(max_iterations, 0, "Max iterations must be positive");
        self.max_iterations = max_iterations;
        self
    }

    #[cfg(any(
        feature = "cpu_backend",
        feature = "opencl_backend",
        feature = "vulkan_backend"
    ))]
    #[allow(clippy::cast_precision_loss)] // loss of precision is acceptable
    pub(crate) fn view_width(&self) -> f32 {
        self.view_height * (self.image_size[0] as f32) / (self.image_size[1] as f32)
    }
}
