//! CPU backend for Julia set rendering.

use arithmetic_parser::BinaryOp;
use num_complex::Complex32;
use rayon::prelude::*;

use std::collections::HashMap;

use crate::{Backend, Evaluated, Function, ImageBuffer, Params};

#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;

impl Backend for Cpu {
    type Error = anyhow::Error;
    type Program = Program;

    fn create_program(&self, function: &Function) -> Result<Self::Program, Self::Error> {
        Ok(Program::new(function.to_owned()))
    }

    fn render(&self, program: &Self::Program, params: &Params) -> Result<ImageBuffer, Self::Error> {
        Ok(program.render(params))
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
        Evaluated::FunctionCall { .. } => unimplemented!(),
    }
}

const MAX_ITERATIONS: usize = 100;

fn smoothstep(low_bound: f32, high_bound: f32, x: f32) -> f32 {
    let clamped_x = if x < low_bound {
        low_bound
    } else if x > high_bound {
        high_bound
    } else {
        x
    };

    clamped_x * clamped_x * (3.0 - 2.0 * clamped_x)
}

#[derive(Debug, Clone, Copy)]
struct CpuParams {
    image_size: [u32; 2],
    image_size_f32: [f32; 2],
    view_size: [f32; 2],
    view_center: Complex32,
    inf_distance_sq: f32,
}

impl CpuParams {
    fn new(params: &Params) -> Self {
        Self {
            image_size: params.image_size,
            image_size_f32: [params.image_size[0] as f32, params.image_size[1] as f32],
            view_size: [params.view_width(), params.view_height],
            view_center: Complex32::new(params.view_center[0], params.view_center[1]),
            inf_distance_sq: params.inf_distance * params.inf_distance,
        }
    }

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

#[derive(Debug)]
pub struct Program {
    function: Function,
}

impl Program {
    fn new(function: Function) -> Self {
        Self { function }
    }

    fn compute(&self, z: Complex32) -> Complex32 {
        let mut variables = HashMap::new();
        variables.insert("z", z);

        for (var_name, expr) in self.function.assignments() {
            let expr = eval(expr, &variables);
            variables.insert(var_name, expr);
        }
        eval(self.function.return_value(), &variables)
    }

    fn compute_row(&self, params: CpuParams, pixel_row: u32) -> Vec<u8> {
        let [image_width, _] = params.image_size;

        let pixels = (0..image_width).map(|pixel_col| {
            let mut z = params.map_pixel(pixel_row, pixel_col);
            let mut iter = MAX_ITERATIONS;

            for i in 0..MAX_ITERATIONS {
                z = self.compute(z);
                if z.is_nan() || z.is_infinite() || z.norm_sqr() > params.inf_distance_sq {
                    iter = i;
                    break;
                }
            }

            let color = iter as f32 / MAX_ITERATIONS as f32;
            let color = smoothstep(0.0, 1.0, 1.0 - color);
            (color * 255.0).round() as u8
        });
        pixels.collect()
    }

    fn render(&self, params: &Params) -> ImageBuffer {
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
        ImageBuffer::from_raw(width, height, buffer).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;

    fn assert_close(x: Complex32, y: Complex32) {
        assert!((x.re - y.re).abs() <= f32::EPSILON, "{:?}, {:?}", x, y);
        assert!((x.im - y.im).abs() <= f32::EPSILON, "{:?}, {:?}", x, y);
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
    fn compute() {
        let program = Program::new(Function::new("z * z + 0.5i").unwrap());
        assert_eq!(
            program.compute(Complex32::new(0.0, 0.0)),
            Complex32::new(0.0, 0.5)
        );
        assert_eq!(
            program.compute(Complex32::new(1.0, 0.0)),
            Complex32::new(1.0, 0.5)
        );
        assert_eq!(
            program.compute(Complex32::new(-1.0, 0.0)),
            Complex32::new(1.0, 0.5)
        );
        assert_eq!(
            program.compute(Complex32::new(0.0, 1.0)),
            Complex32::new(-1.0, 0.5)
        );
    }

    #[test]
    fn compute_does_not_panic() {
        let program = Program::new(Function::new("1.0 / z + 0.5i").unwrap());
        let z = program.compute(Complex32::new(0.0, 0.0));
        assert!(z.is_nan());
    }
}
