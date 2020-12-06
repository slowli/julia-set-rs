use lazy_static::lazy_static;
use ocl::{
    builders::BuildOpt, flags, prm::Float2, Buffer, Context, Device, OclPrm, Platform, ProQue,
    Queue,
};

use std::sync::Mutex;

use crate::{compiler::Compiler, Backend, Function, ImageBuffer, Render, Params};

const PROGRAM: &str = include_str!("program.cl");

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCl;

impl Backend<&Function> for OpenCl {
    type Error = ocl::Error;
    type Program = OpenClProgram;

    fn create_program(function: &Function) -> Result<Self::Program, Self::Error> {
        let compiled = Compiler::for_ocl().compile(function);
        OpenClProgram::new(compiled)
    }
}

#[derive(Debug)]
pub struct OpenClProgram {
    inner: ProQue,
}

impl OpenClProgram {
    fn new(compiled: String) -> ocl::Result<Self> {
        let mut program_builder = ocl::Program::builder();
        let define = BuildOpt::IncludeDefine {
            ident: "COMPUTE(z)".to_owned(),
            val: compiled,
        };
        program_builder.bo(define).source(PROGRAM);

        // For some reason, certain OpenCL implementations (e.g., POCL) do not work well
        // when the list of devices for a platform is queried from multiple threads.
        // Hence, we introduce a `Mutex` to serialize these calls.
        lazy_static! {
            static ref MUTEX: Mutex<()> = Mutex::new(());
        }
        let (platform, device) = {
            let _lock = MUTEX.lock().ok();
            let platform = Platform::first()?;
            (platform, Device::first(platform)?)
        };

        let context = Context::builder()
            .platform(platform)
            .devices(&device)
            .build()?;
        let inner = ProQue::new(
            context.clone(),
            Queue::new(&context, device, None)?,
            program_builder.build(&context)?,
            None::<usize>,
        );
        Ok(Self { inner })
    }
}

impl Render for OpenClProgram {
    type Error = ocl::Error;

    fn render(&self, params: &Params) -> Result<ImageBuffer, Self::Error> {
        let pixels = params.image_size[0]
            .checked_mul(params.image_size[1])
            .expect("Overflow in image dimensions");
        let buffer: Buffer<u8> = Buffer::builder()
            .queue(self.inner.queue().clone())
            .len(pixels)
            .flags(flags::MEM_WRITE_ONLY | flags::MEM_HOST_READ_ONLY)
            .build()?;

        let cl_params = ClParams {
            view_center: Float2::new(params.view_center[0], params.view_center[1]),
            view_size: Float2::new(params.view_width(), params.view_height),
            inf_distance_sq: params.inf_distance * params.inf_distance,
        };
        let kernel = self
            .inner
            .kernel_builder("julia")
            .arg_named("output", &buffer)
            .arg_named("params", cl_params)
            .build()?;

        let command = kernel.cmd().global_work_size(params.image_size);
        unsafe { command.enq()? };

        let mut image = ImageBuffer::new(params.image_size[0], params.image_size[1]);
        buffer.read(&mut *image).enq()?;
        Ok(image)
    }
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ClParams {
    view_center: Float2,
    view_size: Float2,
    inf_distance_sq: f32,
}

// Safety ensured by the same alignment here and in OCL code.
unsafe impl OclPrm for ClParams {}
