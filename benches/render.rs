use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use num_complex::Complex32;

use std::fmt;

use julia_set::{Backend, Cpu, Function, Params, Render};

const IMAGE_DIMENSIONS: [u32; 2] = [1024, 1024];

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

    fn render_cubic(self, bencher: &mut Bencher<'_>) {
        let params = Params::new(IMAGE_DIMENSIONS, 2.5).with_infinity_distance(2.5);

        match self {
            Self::Cpu => {
                let function = |z: Complex32| z * z * z - 0.39;
                let program = Cpu.create_program(function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            Self::DynCpu => {
                let function: Function = "z * z * z - 0.39".parse().unwrap();
                let program = Cpu.create_program(&function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            #[cfg(feature = "opencl_backend")]
            Self::OpenCl => {
                let function: Function = "z * z * z - 0.39".parse().unwrap();
                let program = julia_set::OpenCl.create_program(&function).unwrap();
                bencher.iter(|| program.render(&params).unwrap());
            }

            #[cfg(feature = "vulkan_backend")]
            Self::Vulkan => {
                let function: Function = "z * z * z - 0.39".parse().unwrap();
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

fn render_benches(criterion: &mut Criterion) {
    let mut sq_benches = criterion.benchmark_group("cubic");
    sq_benches.sample_size(10);

    for &backend in BackendName::ALL {
        sq_benches.bench_with_input(backend.to_string(), &backend, |bencher, &backend| {
            backend.render_cubic(bencher)
        });
    }
    sq_benches.finish();
}

criterion_group!(benches, render_benches);
criterion_main!(benches);
