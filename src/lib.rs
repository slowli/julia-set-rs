#![warn(missing_debug_implementations)]

pub use crate::function::{EvalError, Evaluated, FnError, Function};

use std::fmt;

#[cfg(any(test, feature = "opencl_backend", feature = "vulkan_backend"))]
mod compiler;
#[cfg(feature = "cpu_backend")]
mod cpu;
mod function;
#[cfg(feature = "opencl_backend")]
mod opencl;
#[cfg(feature = "vulkan_backend")]
mod vulkan;

#[cfg(feature = "cpu_backend")]
pub use crate::cpu::{Cpu, CpuProgram};
#[cfg(feature = "opencl_backend")]
pub use crate::opencl::OpenCl;
#[cfg(feature = "vulkan_backend")]
pub use crate::vulkan::Vulkan;

pub type ImageBuffer = image::GrayImage;

pub trait Backend: Default {
    type Error: fmt::Debug;
    type Program;

    fn create_program(&self, function: &Function) -> Result<Self::Program, Self::Error>;

    fn render(&self, program: &Self::Program, params: &Params) -> Result<ImageBuffer, Self::Error>;
}

#[derive(Debug, Clone)]
pub struct Params {
    view_center: [f32; 2],
    view_height: f32,
    inf_distance: f32,
    image_size: [u32; 2],
}

impl Params {
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

    pub fn with_view_center(mut self, view_center: [f32; 2]) -> Self {
        self.view_center = view_center;
        self
    }

    pub fn with_infinity_distance(mut self, inf_distance: f32) -> Self {
        assert!(inf_distance > 0.0, "`inf_distance` should be positive");
        self.inf_distance = inf_distance;
        self
    }

    #[cfg(any(feature = "opencl_backend", feature = "vulkan_backend"))]
    pub(crate) fn view_width(&self) -> f32 {
        self.view_height * (self.image_size[0] as f32) / (self.image_size[1] as f32)
    }
}

#[derive(Debug)]
pub struct JuliaSet<B: Backend> {
    program: B::Program,
}

impl<B: Backend> JuliaSet<B> {
    pub fn new(function: &Function) -> Result<Self, B::Error> {
        let backend = B::default();
        let program = backend.create_program(function)?;
        Ok(Self { program })
    }

    pub fn render(&self, params: &Params) -> Result<ImageBuffer, B::Error> {
        B::default().render(&self.program, params)
    }
}
