use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use num_complex::Complex32;

use std::fmt;

use julia_set::{Backend, Cpu, Function, Params, Render};

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

criterion_group!(benches, render_benches);
criterion_main!(benches);
