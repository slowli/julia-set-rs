//! CLI for rendering Julia sets to a file.

use anyhow::anyhow;
use structopt::StructOpt;

use std::{path::PathBuf, str::FromStr};

use julia_set::{Backend, Function, ImageBuffer, Params, Render};

const ABOUT: &str = "CLI for rendering a Julia set to a file.";

#[derive(Debug, StructOpt)]
#[structopt(about = ABOUT)]
struct Args {
    /// Output file.
    #[structopt(name = "output", long, short = "o", default_value = "image.png")]
    output: PathBuf,
    /// Rendering backend to use.
    #[structopt(name = "backend", long, short = "b")]
    backend: BackendName,

    /// Size of the image in pixels.
    #[structopt(name = "size", long, short = "s", default_value = "640x480")]
    size: Size,

    /// X coordinate of the image center.
    #[structopt(name = "cx", long, default_value = "0")]
    center_x: f32,
    /// X coordinate of the image center.
    #[structopt(name = "cy", long, default_value = "0")]
    center_y: f32,
    /// Height of the image (in rendered coordinates).
    #[structopt(name = "height", long, short = "h", default_value = "4")]
    view_height: f32,
    /// Infinity distance for the image.
    #[structopt(name = "inf", long, default_value = "3")]
    infinity_distance: f32,

    /// Complex-valued function for the Julia set, for example, "z * z + 0.5 - 0.4i".
    #[structopt(name = "function")]
    function: String,
}

impl Args {
    fn run(self) -> anyhow::Result<()> {
        println!("Running with {:?}", self);

        let params = Params::new([self.size.width, self.size.height], self.view_height)
            .with_view_center([self.center_x, self.center_y])
            .with_infinity_distance(self.infinity_distance);
        let function = Function::new(&self.function)?;
        let image_buffer = self.backend.compile_and_render(&function, &params)?;
        image_buffer.save(&self.output)?;
        Ok(())
    }
}

#[derive(Debug)]
struct Size {
    width: u32,
    height: u32,
}

impl FromStr for Size {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        let parts: Vec<_> = s.split('x').collect();
        match parts.as_slice() {
            [width, height] => Ok(Self {
                width: width.parse()?,
                height: height.parse()?,
            }),
            _ => Err(anyhow!(
                "Size should consist of width and height separated by `x`, e.g., `640x480`"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BackendName {
    #[cfg(feature = "dyn_cpu_backend")]
    Cpu,
    #[cfg(feature = "opencl_backend")]
    OpenCl,
    #[cfg(feature = "vulkan_backend")]
    Vulkan,
}

impl FromStr for BackendName {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            #[cfg(feature = "dyn_cpu_backend")]
            "cpu" => Ok(Self::Cpu),
            #[cfg(feature = "opencl_backend")]
            "opencl" | "ocl" => Ok(Self::OpenCl),
            #[cfg(feature = "vulkan_backend")]
            "vulkan" => Ok(Self::Vulkan),
            _ => Err(anyhow!(
                "Invalid backend name. Use one of `cpu`, `opencl` or `vulkan`"
            )),
        }
    }
}

impl BackendName {
    fn compile_and_render(
        self,
        function: &Function,
        params: &Params,
    ) -> anyhow::Result<ImageBuffer> {
        Ok(match self {
            #[cfg(feature = "dyn_cpu_backend")]
            Self::Cpu => julia_set::Cpu::create_program(function)?.render(params)?,
            #[cfg(feature = "opencl_backend")]
            Self::OpenCl => julia_set::OpenCl::create_program(function)
                .map_err(|e| anyhow!(e))?
                .render(params)
                .map_err(|e| anyhow!(e))?,
            #[cfg(feature = "vulkan_backend")]
            Self::Vulkan => julia_set::Vulkan::create_program(function)?.render(params)?,
        })
    }
}

fn main() -> anyhow::Result<()> {
    Args::from_args().run()
}
