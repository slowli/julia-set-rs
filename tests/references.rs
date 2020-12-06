//! Reference plots for different backends.

use image::{DynamicImage, ImageError};

use std::{env, io, path::Path};

use julia_set::{Backend, Function, ImageBuffer, JuliaSet, Params};

fn generate_image<B: Backend>(function: &str, params: &Params) -> ImageBuffer {
    let function = Function::new(function).unwrap();
    JuliaSet::<B>::new(&function)
        .unwrap()
        .render(params)
        .unwrap()
}

fn compare_to_reference(reference_filename: &str, image: &ImageBuffer) {
    const ROOT_PATH: &str = env!("CARGO_MANIFEST_DIR");
    let reference_path = Path::new(ROOT_PATH)
        .join("tests")
        .join("__snapshots__")
        .join(reference_filename);

    let reference_image = match image::open(&reference_path) {
        Ok(DynamicImage::ImageLuma8(image)) => Some(image),
        Ok(_) => panic!("Unexpected image format"),

        Err(ImageError::IoError(ref io_error)) if io_error.kind() == io::ErrorKind::NotFound => {
            None
        }
        Err(other_error) => panic!("Error opening reference file: {:?}", other_error),
    };

    if let Some(ref reference_image) = reference_image {
        let image_diff = ImageDiff::new(reference_image, &image);
        println!("{}: {:?}", reference_filename, image_diff);
        image_diff.assert_is_sound();
    } else if env::args().nth(1).as_deref() == Some("--snapshot") {
        // Store the reference image.
        image
            .save(&reference_path)
            .expect("Cannot save reference image");
    } else {
        panic!("Snapshot `{}` not found", reference_filename);
    }
}

#[derive(Debug)]
struct ImageDiff {
    /// Percentage of differing image pixels.
    differing_pixels: f32,
    /// Mean difference in pixel luma across all pixels in the image.
    mean_difference: f32,
}

impl ImageDiff {
    const MAX_DIFFERING_PIXELS: f32 = 0.02;
    const MAX_MEAN_DIFFERENCE: f32 = 0.1;

    fn pixel_quantity(image: &ImageBuffer, quantity: u32) -> f32 {
        let pixel_count = image.width() * image.height();
        quantity as f32 / pixel_count as f32
    }

    fn new(expected: &ImageBuffer, actual: &ImageBuffer) -> Self {
        assert_eq!(expected.width(), actual.width());
        assert_eq!(expected.height(), actual.height());

        let mut differing_count = 0_u32;
        let mut total_diff = 0_u32;

        for (expected_pixel, actual_pixel) in expected.pixels().zip(actual.pixels()) {
            let diff = if expected_pixel[0] > actual_pixel[0] {
                expected_pixel[0] - actual_pixel[0]
            } else {
                actual_pixel[0] - expected_pixel[0]
            };

            if diff > 0 {
                differing_count += 1;
                total_diff += diff as u32;
            }
        }

        let differing_pixels = Self::pixel_quantity(&expected, differing_count);
        let mean_difference = Self::pixel_quantity(&expected, total_diff);
        Self {
            differing_pixels,
            mean_difference,
        }
    }

    fn assert_is_sound(&self) {
        assert!(
            self.differing_pixels <= Self::MAX_DIFFERING_PIXELS,
            "{:?}",
            self
        );
        assert!(
            self.mean_difference <= Self::MAX_MEAN_DIFFERENCE,
            "{:?}",
            self
        );
    }
}

mod cubic {
    use super::*;

    const CUBIC_FUNCTION: &str = "z * z * z - 0.39";
    const SNAPSHOT_FILENAME: &str = "cubic.png";

    fn render_params() -> Params {
        Params::new([360, 360], 2.5).with_infinity_distance(2.5)
    }

    #[test]
    #[cfg(feature = "cpu_backend")]
    fn cpu_backend() {
        let image = generate_image::<julia_set::Cpu>(CUBIC_FUNCTION, &render_params());
        compare_to_reference(SNAPSHOT_FILENAME, &image);
    }

    #[test]
    #[cfg(feature = "cpu_backend")]
    fn cpu_backend_with_native_function() {
        let image = julia_set::CpuProgram::new(|z| z * z * z - 0.39).render(&render_params());
        compare_to_reference(SNAPSHOT_FILENAME, &image);
    }

    #[test]
    #[cfg(feature = "opencl_backend")]
    fn vulkan_backend() {
        let image = generate_image::<julia_set::Vulkan>(CUBIC_FUNCTION, &render_params());
        compare_to_reference(SNAPSHOT_FILENAME, &image);
    }

    #[test]
    #[cfg(feature = "vulkan_backend")]
    fn opencl_backend() {
        let image = generate_image::<julia_set::OpenCl>(CUBIC_FUNCTION, &render_params());
        compare_to_reference(SNAPSHOT_FILENAME, &image);
    }
}
