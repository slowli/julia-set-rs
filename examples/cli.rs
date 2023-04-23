//! CLI for rendering Julia sets to a file.

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use image::{Luma, Rgb};

use std::{path::PathBuf, str::FromStr};

use julia_set::{
    transform::{ApplyTransform, Negative, Palette, PixelTransform, Smoothstep},
    Backend, Function, ImageBuffer, Params, Render,
};

type LumaTransform = dyn PixelTransform<Luma<u8>, Output = Luma<u8>>;

const ABOUT: &str = "CLI for rendering a Julia set to a file.";

#[derive(Debug, Parser)]
#[command(author, version, about = ABOUT, long_about = None)]
struct Cli {
    /// Output file for the rendered image.
    #[arg(long, short = 'o', default_value = "image.png")]
    output: PathBuf,
    /// Rendering backend to use.
    #[arg(long, short = 'b', value_enum)]
    backend: BackendName,

    /// Size of the image in pixels.
    #[arg(long, short = 's', default_value = "640x480")]
    size: Size,

    /// X coordinate of the image center.
    #[arg(name = "cx", long, default_value = "0")]
    center_x: f32,
    /// X coordinate of the image center.
    #[arg(name = "cy", long, default_value = "0")]
    center_y: f32,
    /// Height of the image (in rendered coordinates).
    #[arg(name = "height", long, short = 'H', default_value = "4")]
    view_height: f32,
    /// Infinity distance for the image.
    #[arg(name = "inf", long, default_value = "3")]
    infinity_distance: f32,
    /// Max iteration count.
    #[arg(name = "iter", long, default_value = "100")]
    max_iterations: u8,

    /// List of luma transforms.
    #[arg(name = "tr", long, default_value = "")]
    transforms: NamedLumaTransforms,
    /// Palette to use. If not specified, the image will be grayscale.
    #[arg(name = "palette", long, short = 'p')]
    palette: Option<NamedPalette>,

    /// Complex-valued function for the Julia set, for example, "z * z + 0.5 - 0.4i".
    #[arg(name = "function")]
    function: String,
}

impl Cli {
    fn luma_transform(&self) -> Box<LumaTransform> {
        match self.transforms.0.as_slice() {
            [] => Box::new(()),
            [x] => x.to_pixel_transform(),
            [x, rest @ ..] => rest
                .iter()
                .map(|y| y.to_pixel_transform())
                .fold(x.to_pixel_transform(), |acc, y| Box::new((acc, y))),
        }
    }

    fn run(self) -> anyhow::Result<()> {
        println!("Running with {self:?}");

        let params = Params::new([self.size.width, self.size.height], self.view_height)
            .with_view_center([self.center_x, self.center_y])
            .with_infinity_distance(self.infinity_distance)
            .with_max_iterations(self.max_iterations);
        let function = self.function.parse()?;
        let iterations = self.backend.compile_and_render(&function, &params)?;
        let transformed_luma = iterations.apply(self.luma_transform());

        if let Some(palette) = self.palette {
            transformed_luma
                .apply(palette.to_palette())
                .transform()
                .save(&self.output)?;
        } else {
            transformed_luma.transform().save(&self.output)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "lower")]
enum BackendName {
    #[cfg(feature = "dyn_cpu_backend")]
    Cpu,
    #[cfg(feature = "opencl_backend")]
    #[value(alias("ocl"))]
    OpenCl,
    #[cfg(feature = "vulkan_backend")]
    Vulkan,
}

impl BackendName {
    fn compile_and_render(
        self,
        function: &Function,
        params: &Params,
    ) -> anyhow::Result<ImageBuffer> {
        Ok(match self {
            #[cfg(feature = "dyn_cpu_backend")]
            Self::Cpu => julia_set::Cpu.create_program(function)?.render(params)?,
            #[cfg(feature = "opencl_backend")]
            Self::OpenCl => julia_set::OpenCl
                .create_program(function)
                .map_err(|e| anyhow!(e))?
                .render(params)
                .map_err(|e| anyhow!(e))?,
            #[cfg(feature = "vulkan_backend")]
            Self::Vulkan => julia_set::Vulkan.create_program(function)?.render(params)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum NamedPalette {
    Tree,
    Gold,
    Snow,
}

impl FromStr for NamedPalette {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "tree" => Ok(Self::Tree),
            "gold" => Ok(Self::Gold),
            "snow" => Ok(Self::Snow),
            _ => Err(anyhow!(
                "Invalid palette. Use one of `tree`, `gold` or `snow`"
            )),
        }
    }
}

impl NamedPalette {
    fn to_palette(self) -> Palette<Rgb<u8>> {
        match self {
            Self::Tree => Palette::new(&[
                Rgb([0, 0, 0]),
                Rgb([170, 0, 0]),
                Rgb([136, 204, 0]),
                Rgb([170, 255, 136]),
                Rgb([255, 255, 170]),
                Rgb([255, 255, 255]),
            ]),

            Self::Gold => Palette::new(&[
                Rgb([0, 0, 0]),
                Rgb([187, 187, 0]),
                Rgb([255, 187, 0]),
                Rgb([255, 255, 0]),
                Rgb([255, 255, 136]),
                Rgb([255, 255, 255]),
            ]),

            Self::Snow => {
                Palette::new(&[Rgb([0, 0, 0]), Rgb([136, 170, 255]), Rgb([255, 255, 255])])
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NamedLumaTransform {
    Negative,
    Smoothstep,
}

impl FromStr for NamedLumaTransform {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "neg" | "negative" => Ok(Self::Negative),
            "smooth" | "smoothstep" => Ok(Self::Smoothstep),
            _ => Err(anyhow!("Invalid transform: {s}")),
        }
    }
}

impl NamedLumaTransform {
    fn to_pixel_transform(self) -> Box<LumaTransform> {
        match self {
            Self::Negative => Box::new(Negative),
            Self::Smoothstep => Box::new(Smoothstep),
        }
    }
}

#[derive(Debug, Clone)]
struct NamedLumaTransforms(Vec<NamedLumaTransform>);

impl FromStr for NamedLumaTransforms {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            Ok(Self(vec![]))
        } else {
            trimmed
                .split(',')
                .map(|part| part.trim().parse::<NamedLumaTransform>())
                .collect::<anyhow::Result<Vec<_>>>()
                .map(Self)
        }
    }
}

fn main() -> anyhow::Result<()> {
    Cli::parse().run()
}
