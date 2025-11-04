//! Benchmarks for parsing, compiling and rendering Julia sets on all supported backends.

use std::fmt;

use criterion::{Bencher, Criterion, criterion_group, criterion_main};
use julia_set::{Backend, Cpu, Function, Params, Render};
use num_complex::Complex32;

const SAMPLE_SIZE: usize = 10;
const IMAGE_SIZE: [u32; 2] = [640, 360];

#[derive(Debug, Clone, Copy)]
enum BackendName {
    Cpu,
    DynCpu,
    #[cfg(feature = "opencl_backend")]
    OpenCl,
    #[cfg(feature = "vulkan_backend")]
    Vulkan,
}

impl BackendName {
    const ALL: &'static [Self] = &[
        Self::Cpu,
        Self::DynCpu,
        #[cfg(feature = "opencl_backend")]
        Self::OpenCl,
        #[cfg(feature = "vulkan_backend")]
        Self::Vulkan,
    ];

    fn render_function(
        self,
        bencher: &mut Bencher<'_>,
        params: Params,
        native_fn: fn(Complex32) -> Complex32,
        fn_str: &str,
    ) {
        match self {
            Self::Cpu => {
                let program = Cpu.create_program(native_fn).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            Self::DynCpu => {
                let function: Function = fn_str.parse().unwrap();
                let program = Cpu.create_program(&function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            #[cfg(feature = "opencl_backend")]
            Self::OpenCl => {
                let function: Function = fn_str.parse().unwrap();
                let program = julia_set::OpenCl.create_program(&function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            #[cfg(feature = "vulkan_backend")]
            Self::Vulkan => {
                let function: Function = fn_str.parse().unwrap();
                let program = julia_set::Vulkan.create_program(&function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }
        }
    }
}

impl fmt::Display for BackendName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::DynCpu => "dyn_cpu",
            #[cfg(feature = "opencl_backend")]
            Self::OpenCl => "opencl",
            #[cfg(feature = "vulkan_backend")]
            Self::Vulkan => "vulkan",
        })
    }
}

fn parse_benches(criterion: &mut Criterion) {
    criterion
        .bench_function("cubic/parse", |bencher| {
            bencher.iter(|| "z * z * z - 0.39".parse::<Function>().unwrap())
        })
        .bench_function("flower/parse", |bencher| {
            bencher.iter(|| "0.8*z + z/atanh(z^-4)".parse::<Function>().unwrap())
        })
        .bench_function("hills/parse", |bencher| {
            bencher.iter(|| {
                "1i * acosh(cosh(1i * z) - arg(z)^-2) - 0.05 + 0.05i"
                    .parse::<Function>()
                    .unwrap()
            })
        });
}

fn compile_benches(criterion: &mut Criterion) {
    #[cfg(feature = "opencl_backend")]
    criterion
        .benchmark_group("opencl")
        .bench_function("compile_cubic", |bencher| {
            let function: Function = "z * z * z - 0.39".parse::<Function>().unwrap();
            bencher.iter(|| julia_set::OpenCl.create_program(&function).unwrap());
        })
        .sample_size(SAMPLE_SIZE);

    #[cfg(feature = "vulkan_backend")]
    criterion
        .benchmark_group("vulkan")
        .bench_function("compile_cubic", |bencher| {
            let function: Function = "z * z * z - 0.39".parse::<Function>().unwrap();
            bencher.iter(|| julia_set::Vulkan.create_program(&function).unwrap());
        })
        .sample_size(SAMPLE_SIZE);
}

fn bench_function(
    criterion: &mut Criterion,
    fn_name: &str,
    params: Params,
    native_fn: fn(Complex32) -> Complex32,
    fn_str: &str,
) {
    let mut benches = criterion.benchmark_group(fn_name);
    benches.sample_size(SAMPLE_SIZE);
    for &backend in BackendName::ALL {
        benches.bench_with_input(backend.to_string(), &backend, |bencher, &backend| {
            backend.render_function(bencher, params.clone(), native_fn, fn_str);
        });
    }
    benches.finish();
}

fn render_benches(criterion: &mut Criterion) {
    bench_function(
        criterion,
        "cubic",
        Params::new(IMAGE_SIZE, 2.5).with_infinity_distance(2.5),
        |z| z * z * z - 0.39,
        "z * z * z - 0.39",
    );

    bench_function(
        criterion,
        "flower",
        Params::new(IMAGE_SIZE, 2.0).with_infinity_distance(10.0),
        |z| z * 0.8 + z / z.powi(-4).atanh(),
        "0.8*z + z/atanh(z^-4)",
    );

    bench_function(
        criterion,
        "hills",
        Params::new(IMAGE_SIZE, 8.0)
            .with_view_center([-9.41, 0.0])
            .with_infinity_distance(5.0),
        |z| {
            Complex32::i() * ((Complex32::i() * z).cosh() - z.arg().powi(-2)).acosh()
                + Complex32::new(-0.05, 0.05)
        },
        "1i * acosh(cosh(1i * z) - arg(z)^-2) - 0.05 + 0.05i",
    );
}

criterion_group!(benches, parse_benches, compile_benches, render_benches);
criterion_main!(benches);
