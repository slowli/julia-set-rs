//! CPU backend for Julia set rendering.

use num_complex::Complex32;
use rayon::prelude::*;

use std::convert::Infallible;

use crate::{Backend, ImageBuffer, Params, Render};

/// Backend that uses CPU for computations.
///
/// The current implementation is based on the [`rayon`] crate.
///
/// [`rayon`]: https://crates.io/crates/rayon
#[cfg_attr(docsrs, doc(cfg(feature = "cpu_backend")))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;

impl<F: ComputePoint> Backend<F> for Cpu {
    type Error = Infallible;
    type Program = CpuProgram<F>;

    fn create_program(&self, function: F) -> Result<Self::Program, Self::Error> {
        Ok(CpuProgram::new(function))
    }
}

#[derive(Debug, Clone, Copy)]
struct CpuParams {
    image_size: [u32; 2],
    image_size_f32: [f32; 2],
    view_size: [f32; 2],
    view_center: Complex32,
    inf_distance_sq: f32,
    max_iterations: u8,
}

impl CpuParams {
    #[allow(clippy::cast_precision_loss)] // The loss of precision is acceptable
    fn new(params: &Params) -> Self {
        Self {
            image_size: params.image_size,
            image_size_f32: [params.image_size[0] as f32, params.image_size[1] as f32],
            view_size: [params.view_width(), params.view_height],
            view_center: Complex32::new(params.view_center[0], params.view_center[1]),
            inf_distance_sq: params.inf_distance * params.inf_distance,
            max_iterations: params.max_iterations,
        }
    }

    #[allow(clippy::cast_precision_loss)] // The loss of precision is acceptable
    fn map_pixel(self, pixel_row: u32, pixel_col: u32) -> Complex32 {
        let [width, height] = self.image_size_f32;
        let [view_width, view_height] = self.view_size;

        let re = (pixel_col as f32 + 0.5) / width;
        let re = (re - 0.5) * view_width;
        let im = (pixel_row as f32 + 0.5) / height;
        let im = (0.5 - im) * view_height;
        Complex32::new(re, im) + self.view_center
    }
}

/// Complex-valued function of a single variable.
#[cfg_attr(docsrs, doc(cfg(feature = "cpu_backend")))]
pub trait ComputePoint: Sync {
    /// Computes the function value at the specified point.
    fn compute_point(&self, z: Complex32) -> Complex32;
}

/// Programs output by the [`Cpu`] backend. Come in two varieties depending on the type param:
/// native closures, or interpreted [`Function`](crate::Function)s.
#[cfg_attr(docsrs, doc(cfg(feature = "cpu_backend")))]
#[derive(Debug)]
pub struct CpuProgram<F> {
    function: F,
}

impl<F: Fn(Complex32) -> Complex32 + Sync> ComputePoint for F {
    fn compute_point(&self, z: Complex32) -> Complex32 {
        self(z)
    }
}

impl<F: ComputePoint> CpuProgram<F> {
    fn new(function: F) -> Self {
        Self { function }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn compute_row(&self, params: CpuParams, pixel_row: u32) -> Vec<u8> {
        let [image_width, _] = params.image_size;

        let pixels = (0..image_width).map(|pixel_col| {
            let mut z = params.map_pixel(pixel_row, pixel_col);
            let mut iter = params.max_iterations;

            for i in 0..params.max_iterations {
                z = self.function.compute_point(z);
                if z.is_nan() || z.is_infinite() || z.norm_sqr() > params.inf_distance_sq {
                    iter = i;
                    break;
                }
            }

            let color = f32::from(iter) / f32::from(params.max_iterations);
            (color * 255.0).round() as u8 // cannot truncate or lose sign by construction
        });
        pixels.collect()
    }
}

impl<F: ComputePoint> Render for CpuProgram<F> {
    type Error = Infallible;

    fn render(&self, params: &Params) -> Result<ImageBuffer, Self::Error> {
        let [width, height] = params.image_size;
        let pixel_size = (width * height) as usize;
        let params = CpuParams::new(params);

        let buffer: Vec<u8> = (0..height)
            .into_par_iter()
            .fold(
                || Vec::with_capacity(pixel_size),
                |mut buffer, pixel_row| {
                    let line = self.compute_row(params, pixel_row);
                    buffer.extend_from_slice(&line);
                    buffer
                },
            )
            .flatten()
            .collect();
        Ok(ImageBuffer::from_raw(width, height, buffer).unwrap())
    }
}

#[cfg(feature = "dyn_cpu_backend")]
mod dynamic {
    use arithmetic_parser::BinaryOp;
    use num_complex::Complex32;

    use std::{collections::HashMap, convert::Infallible};

    use super::{ComputePoint, Cpu, CpuProgram};
    use crate::{Backend, Function, function::Evaluated};

    impl Backend<&Function> for Cpu {
        type Error = Infallible;
        type Program = CpuProgram<Function>;

        fn create_program(&self, function: &Function) -> Result<Self::Program, Self::Error> {
            Ok(CpuProgram::new(function.clone()))
        }
    }

    fn eval(expr: &Evaluated, variables: &HashMap<&str, Complex32>) -> Complex32 {
        match expr {
            Evaluated::Variable(s) => variables[s.as_str()],
            Evaluated::Value(val) => *val,
            Evaluated::Negation(inner) => -eval(inner, variables),
            Evaluated::Binary { op, lhs, rhs } => {
                let lhs_value = eval(lhs, variables);
                let rhs_value = eval(rhs, variables);
                match op {
                    BinaryOp::Add => lhs_value + rhs_value,
                    BinaryOp::Sub => lhs_value - rhs_value,
                    BinaryOp::Mul => lhs_value * rhs_value,
                    BinaryOp::Div => lhs_value / rhs_value,
                    BinaryOp::Power => lhs_value.powc(rhs_value),
                    _ => unreachable!(),
                }
            }
            Evaluated::FunctionCall { function, arg } => {
                let evaluated_arg = eval(arg, variables);
                function.eval(evaluated_arg)
            }
        }
    }

    impl ComputePoint for Function {
        fn compute_point(&self, z: Complex32) -> Complex32 {
            let mut variables = HashMap::new();
            variables.insert("z", z);

            for (var_name, expr) in self.assignments() {
                let expr = eval(expr, &variables);
                variables.insert(var_name, expr);
            }
            eval(self.return_value(), &variables)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(x: Complex32, y: Complex32) {
        assert!((x.re - y.re).abs() <= f32::EPSILON, "{x:?}, {y:?}");
        assert!((x.im - y.im).abs() <= f32::EPSILON, "{x:?}, {y:?}");
    }

    #[test]
    fn mapping_pixels() {
        let params = Params::new([100, 100], 1.0);
        let params = CpuParams::new(&params);
        assert_close(params.map_pixel(0, 0), Complex32::new(-0.495, 0.495));
        assert_close(params.map_pixel(0, 50), Complex32::new(0.005, 0.495));
        assert_close(params.map_pixel(0, 100), Complex32::new(0.505, 0.495));
        assert_close(params.map_pixel(50, 0), Complex32::new(-0.495, -0.005));
        assert_close(params.map_pixel(50, 50), Complex32::new(0.005, -0.005));
        assert_close(params.map_pixel(50, 100), Complex32::new(0.505, -0.005));
        assert_close(params.map_pixel(100, 0), Complex32::new(-0.495, -0.505));
        assert_close(params.map_pixel(100, 50), Complex32::new(0.005, -0.505));
        assert_close(params.map_pixel(100, 100), Complex32::new(0.505, -0.505));
    }

    #[test]
    #[cfg(feature = "dyn_cpu_backend")]
    fn compute() {
        use crate::Function;

        let program: Function = "z * z + 0.5i".parse().unwrap();
        assert_eq!(
            program.compute_point(Complex32::new(0.0, 0.0)),
            Complex32::new(0.0, 0.5)
        );
        assert_eq!(
            program.compute_point(Complex32::new(1.0, 0.0)),
            Complex32::new(1.0, 0.5)
        );
        assert_eq!(
            program.compute_point(Complex32::new(-1.0, 0.0)),
            Complex32::new(1.0, 0.5)
        );
        assert_eq!(
            program.compute_point(Complex32::new(0.0, 1.0)),
            Complex32::new(-1.0, 0.5)
        );
    }

    #[test]
    #[cfg(feature = "dyn_cpu_backend")]
    fn compute_does_not_panic() {
        use crate::Function;

        let program: Function = "1.0 / z + 0.5i".parse().unwrap();
        let z = program.compute_point(Complex32::new(0.0, 0.0));
        assert!(z.is_nan());
    }
}
