//! Post-processing transforms on Julia set images.
//!
//! These transforms are not meant to be exhaustive; rather, they represent some functionality
//! lacking from [`image`] and [`imageproc`] crates. You can absolutely use the transforms from
//! these crates together with (or instead of) transforms defined here.
//!
//! [`image`]: https://crates.io/crates/image
//! [`imageproc`]: https://crates.io/crates/imageproc
//!
//! # Examples
//!
//!
//! ```
//! # use image::GrayImage;
//! use julia_set::transform::{ApplyTransform, Negative, Smoothstep};
//!
//! let image: GrayImage = // ...
//! #    GrayImage::from_raw(10, 10, vec![0_u8; 100]).unwrap();
//! let transformed_image = image.apply(Negative).apply(Smoothstep).transform();
//! ```

use image::{ImageBuffer, Luma, Pixel};

/// Pixel-wise transform.
///
/// The function defined by the transform is applied separately to each pixel of the source image.
/// The transform may change the type of pixels (e.g., transform from grayscale to a color image),
/// or may leave it intact.
pub trait PixelTransform<Pix: Pixel> {
    /// Pixel type for the output image.
    type Output: Pixel + 'static;

    /// Performs transform on a single pixel.
    fn transform_pixel(&self, pixel: Pix) -> Self::Output;
}

/// No-op transform.
impl<Pix: Pixel + 'static> PixelTransform<Pix> for () {
    type Output = Pix;

    #[inline]
    fn transform_pixel(&self, pixel: Pix) -> Self::Output {
        pixel
    }
}

impl<Pix, O> PixelTransform<Pix> for Box<dyn PixelTransform<Pix, Output = O>>
where
    Pix: Pixel,
    O: Pixel + 'static,
{
    type Output = O;

    #[inline]
    fn transform_pixel(&self, pixel: Pix) -> Self::Output {
        (**self).transform_pixel(pixel)
    }
}

/// Composition of two transforms. The transforms are applied in the same order they are listed
/// in a tuple.
impl<F, G, Pix: Pixel> PixelTransform<Pix> for (F, G)
where
    F: PixelTransform<Pix>,
    G: PixelTransform<F::Output>,
{
    type Output = G::Output;

    #[inline]
    fn transform_pixel(&self, pixel: Pix) -> Self::Output {
        self.1.transform_pixel(self.0.transform_pixel(pixel))
    }
}

/// Transform that negates the value of each pixel of the input image.
#[derive(Debug, Clone, Copy, Default)]
pub struct Negative;

impl PixelTransform<Luma<u8>> for Negative {
    type Output = Luma<u8>;

    #[inline]
    fn transform_pixel(&self, pixel: Luma<u8>) -> Self::Output {
        Luma([u8::max_value() - pixel[0]])
    }
}

/// Transform that smooths the value of each pixel of the input image using cubic
/// Hermite interpolation (aka `smoothstep` from GLSL / OpenCL).
#[derive(Debug, Clone, Copy, Default)]
pub struct Smoothstep;

impl PixelTransform<Luma<u8>> for Smoothstep {
    type Output = Luma<u8>;

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn transform_pixel(&self, pixel: Luma<u8>) -> Self::Output {
        let clamped_x = f32::from(pixel[0]) / 255.0;
        let output = clamped_x * clamped_x * (3.0 - 2.0 * clamped_x);
        Luma([(output * 255.0).round() as u8])
    }
}

/// Transform that colorizes the image using a palette.
#[derive(Debug, Clone)]
pub struct Palette<T> {
    pixels: [T; 256],
}

impl<T> Palette<T>
where
    T: Pixel<Subpixel = u8> + 'static,
{
    /// Creates a palette based on the provided color stops. The provided `colors` are distributed
    /// at equal distances along the palette. Intermediate colors are obtained
    /// by linear interpolation between the two nearest color stops.
    ///
    /// # Panics
    ///
    /// - Panics if the number of `colors` is less than 2 or greater than 256.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn new(colors: &[T]) -> Self {
        assert!(colors.len() >= 2, "palette must contain at least 2 colors");
        assert!(
            colors.len() <= 256,
            "palette cannot contain more than 256 colors"
        );
        let len_scale = (colors.len() - 1) as f32;

        let mut pixels = [T::from_channels(0, 0, 0, 0); 256];
        for (i, pixel) in pixels.iter_mut().enumerate() {
            let float_i = i as f32 / 255.0 * len_scale;

            let mut prev_color_idx = float_i as usize; // floors as expected
            if prev_color_idx == colors.len() - 1 {
                prev_color_idx -= 1;
            }
            debug_assert!(prev_color_idx + 1 < colors.len());

            let prev_color = colors[prev_color_idx].channels();
            let next_color = colors[prev_color_idx + 1].channels();
            let blend_factor = float_i - prev_color_idx as f32;
            debug_assert!(blend_factor >= 0.0 && blend_factor <= 1.0);

            let mut blended_channels = [0_u8; 4];
            let channel_count = T::CHANNEL_COUNT as usize;
            for (ch, blended_channel) in blended_channels[..channel_count].iter_mut().enumerate() {
                let blended = f32::from(prev_color[ch]) * (1.0 - blend_factor)
                    + f32::from(next_color[ch]) * blend_factor;
                *blended_channel = blended.round() as u8;
            }
            *pixel = *T::from_slice(&blended_channels[..channel_count]);
        }

        Self { pixels }
    }
}

impl<Pix: Pixel + 'static> PixelTransform<Luma<u8>> for Palette<Pix> {
    type Output = Pix;

    #[inline]
    fn transform_pixel(&self, pixel: Luma<u8>) -> Self::Output {
        self.pixels[pixel[0] as usize]
    }
}

/// Trait allowing to lazily apply one or more [`PixelTransform`]s to an image.
///
/// This trait is implemented for [`ImageBuffer`]s and for [`ImageAndTransform`]s, allowing
/// to chain transforms.
pub trait ApplyTransform<Pix: Pixel, F> {
    /// Combined transform after applying `transform`.
    type CombinedTransform: PixelTransform<Pix>;
    /// Appends `transform` to the list of transforms to be performed on the source image.
    fn apply(self, transform: F) -> ImageAndTransform<Pix, Self::CombinedTransform>;
}

/// Source image together with one or more [`PixelTransform`]s. Transforms are applied lazily,
/// once [`Self::transform()`] is called.
#[derive(Debug)]
pub struct ImageAndTransform<Pix, F>
where
    Pix: Pixel,
{
    source_image: ImageBuffer<Pix, Vec<Pix::Subpixel>>,
    transform: F,
}

impl<Pix, F> ImageAndTransform<Pix, F>
where
    Pix: Pixel + Copy + 'static,
    F: PixelTransform<Pix>,
    <F::Output as Pixel>::Subpixel: 'static,
{
    /// Applies transforms accumulated in this object to the source image.
    pub fn transform(&self) -> ImageBuffer<F::Output, Vec<<F::Output as Pixel>::Subpixel>> {
        let mut output = ImageBuffer::new(self.source_image.width(), self.source_image.height());

        let output_iter = self
            .source_image
            .enumerate_pixels()
            .map(|(x, y, pixel)| (x, y, self.transform.transform_pixel(*pixel)));
        for (x, y, out_pixel) in output_iter {
            output[(x, y)] = out_pixel;
        }
        output
    }
}

impl<Pix, F> ApplyTransform<Pix, F> for ImageBuffer<Pix, Vec<Pix::Subpixel>>
where
    Pix: Pixel,
    F: PixelTransform<Pix>,
{
    type CombinedTransform = F;

    fn apply(self, transform: F) -> ImageAndTransform<Pix, F> {
        ImageAndTransform {
            source_image: self,
            transform,
        }
    }
}

impl<Pix, F, G> ApplyTransform<Pix, G> for ImageAndTransform<Pix, F>
where
    Pix: Pixel,
    F: PixelTransform<Pix>,
    G: PixelTransform<F::Output>,
{
    type CombinedTransform = (F, G);

    fn apply(self, transform: G) -> ImageAndTransform<Pix, (F, G)> {
        ImageAndTransform {
            source_image: self.source_image,
            transform: (self.transform, transform),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use image::{GrayImage, Rgb};

    #[test]
    fn simple_transform() {
        let image = GrayImage::from_fn(100, 100, |x, y| Luma::from([(x + y) as u8]));
        let image = image.apply(Negative).apply(Smoothstep).transform();
        for (x, y, pix) in image.enumerate_pixels() {
            let negated = (255 - x - y) as f32 / 255.0;
            let smoothed = negated * negated * (3.0 - 2.0 * negated);
            let expected_pixel = (smoothed * 255.0).round() as u8;
            assert_eq!(pix[0], expected_pixel);
        }
    }

    #[test]
    fn palette_basics() {
        let palette = Palette::new(&[Rgb([0, 255, 0]), Rgb([255, 255, 255])]);
        for (i, &pixel) in palette.pixels.iter().enumerate() {
            assert_eq!(pixel, Rgb([i as u8, 255, i as u8]));
        }
    }
}
