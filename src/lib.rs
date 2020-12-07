//! Julia set rendering.
//!
//! FIXME: Definition
//! FIXME: Features
//! FIXME: Example

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs, missing_debug_implementations)]

use std::fmt;

#[cfg(feature = "arithmetic-parser")]
pub use crate::function::{FnError, Function};
#[cfg(feature = "cpu_backend")]
pub use crate::cpu::{ComputePoint, Cpu, CpuProgram};
#[cfg(feature = "opencl_backend")]
pub use crate::opencl::{OpenCl, OpenClProgram};
#[cfg(feature = "vulkan_backend")]
pub use crate::vulkan::{Vulkan, VulkanProgram};

#[cfg(feature = "arithmetic-parser")]
mod compiler;
#[cfg(feature = "cpu_backend")]
mod cpu;
#[cfg(feature = "arithmetic-parser")]
mod function;
#[cfg(feature = "opencl_backend")]
mod opencl;
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
/// - **Image dimensions** (in pixels)
/// - **View dimensions** and **view center** determine the rendered area. (Only the view height
///   is specified explicitly; the width is inferred from the height and
///   the image dimension ratio.)
/// - **Infinity distance** (see the [Julia set definition] for more details)
#[derive(Debug, Clone)]
pub struct Params {
    view_center: [f32; 2],
    view_height: f32,
    inf_distance: f32,
    image_size: [u32; 2],
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

    pub(crate) fn view_width(&self) -> f32 {
        self.view_height * (self.image_size[0] as f32) / (self.image_size[1] as f32)
    }
}
