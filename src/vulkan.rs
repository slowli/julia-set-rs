//! GLSL / Vulkan backend for Julia sets.

use anyhow::format_err;
use shaderc::{CompilationArtifact, CompileOptions, OptimizationLevel, ShaderKind};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::AutoCommandBufferBuilder,
    descriptor::{
        descriptor::{DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages},
        descriptor_set::{PersistentDescriptorSet, UnsafeDescriptorSetLayout},
        pipeline_layout::{PipelineLayout, PipelineLayoutDesc, PipelineLayoutDescPcRange},
        PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily},
    pipeline::{shader::ShaderModule, ComputePipeline},
    sync::{self, GpuFuture},
};

use std::{ffi::CStr, iter, slice, sync::Arc};

use crate::{compiler::Compiler, Backend, Function, ImageBuffer, Params, Render};

const PROGRAM: &str = include_str!(concat!(env!("OUT_DIR"), "/program.glsl"));

const LOCAL_WORKGROUP_SIZES: [u32; 2] = [16, 16];

fn compile_shader(function: &str) -> shaderc::Result<CompilationArtifact> {
    let mut compiler = shaderc::Compiler::new().ok_or_else(|| {
        shaderc::Error::NullResultObject("Cannot initialize `shaderc` compiler".to_owned())
    })?;
    let mut options = CompileOptions::new().ok_or_else(|| {
        shaderc::Error::NullResultObject("Cannot initialize `shaderc` compiler options".to_owned())
    })?;
    options.add_macro_definition("COMPUTE", Some(function));
    options.set_optimization_level(OptimizationLevel::Performance);
    compiler.compile_into_spirv(
        PROGRAM,
        ShaderKind::Compute,
        "program.glsl",
        "main",
        Some(&options),
    )
}

/// Hand-written layout spec for the compute GLSL shader.
#[derive(Debug, Clone, Copy)]
struct Layout;

unsafe impl PipelineLayoutDesc for Layout {
    fn num_sets(&self) -> usize {
        1
    }

    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        if set == 0 {
            Some(2)
        } else {
            None
        }
    }

    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        let stages = ShaderStages {
            compute: true,
            ..ShaderStages::none()
        };

        match (set, binding) {
            (0, 0) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: true,
                }),
                array_count: 1,
                stages,
                readonly: false,
            }),

            (0, 1) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: false,
                }),
                array_count: 1,
                stages,
                readonly: true,
            }),

            _ => None,
        }
    }

    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    fn push_constants_range(&self, _num: usize) -> Option<PipelineLayoutDescPcRange> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct VulkanParams {
    view_center: [f32; 2],
    view_size: [f32; 2],
    image_size: [u32; 2],
    inf_distance_sq: f32,
    max_iterations: u32,
}

/// Backend based on [Vulkan].
///
/// [Vulkan]: https://www.khronos.org/vulkan/
#[cfg_attr(docsrs, doc(cfg(feature = "vulkan_backend")))]
#[derive(Debug, Clone, Default)]
pub struct Vulkan;

impl Backend<&Function> for Vulkan {
    type Error = anyhow::Error;
    type Program = VulkanProgram;

    fn create_program(&self, function: &Function) -> Result<Self::Program, Self::Error> {
        let compiled = Compiler::for_gl().compile(function);
        VulkanProgram::new(&compiled)
    }
}

/// Program produced by the [`Vulkan`] backend.
#[cfg_attr(docsrs, doc(cfg(feature = "vulkan_backend")))]
#[derive(Debug)]
pub struct VulkanProgram {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline<PipelineLayout<Layout>>>,
    layout: Arc<UnsafeDescriptorSetLayout>,
}

impl VulkanProgram {
    fn new(compiled_function: &str) -> anyhow::Result<Self> {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)?;
        let device = PhysicalDevice::enumerate(&instance)
            .next()
            .ok_or_else(|| format_err!("Physical device not found for instance {:?}", instance))?;
        let queue_family = device
            .queue_families()
            .find(QueueFamily::supports_compute)
            .ok_or_else(|| format_err!("No support of compute shaders on {:?}", device))?;

        let (device, mut queues) = Device::new(
            device,
            device.supported_features(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            iter::once((queue_family, 0.5)),
        )?;
        let queue = queues
            .next()
            .ok_or_else(|| format_err!("Cannot initialize compute queue on device {:?}", device))?;

        let shader = compile_shader(compiled_function)?;
        let shader = unsafe { ShaderModule::from_words(device.clone(), shader.as_binary())? };
        let entry_point_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let entry_point = unsafe { shader.compute_entry_point(entry_point_name, Layout) };

        let pipeline = ComputePipeline::new(device.clone(), &entry_point, &(), None)?;
        let layout = pipeline
            .layout()
            .descriptor_set_layout(0)
            .unwrap() // safe: we know for sure that we have 0-th descriptor set
            .to_owned();

        Ok(Self {
            device,
            queue,
            pipeline: Arc::new(pipeline),
            layout,
        })
    }
}

impl Render for VulkanProgram {
    type Error = anyhow::Error;

    #[allow(clippy::cast_possible_truncation)]
    fn render(&self, params: &Params) -> anyhow::Result<ImageBuffer> {
        // Bind uniforms: the output image buffer and the rendering params.
        let pixel_count = (params.image_size[0] * params.image_size[1]) as usize;
        let image_buffer = unsafe {
            CpuAccessibleBuffer::<[u32]>::uninitialized_array(
                self.device.clone(),
                (pixel_count + 3) / 4,
                BufferUsage {
                    storage_buffer: true,
                    transfer_destination: true,
                    ..BufferUsage::none()
                },
                true,
            )
        }?;

        let gl_params = VulkanParams {
            view_center: params.view_center,
            view_size: [params.view_width(), params.view_height],
            image_size: params.image_size,
            inf_distance_sq: params.inf_distance * params.inf_distance,
            max_iterations: u32::from(params.max_iterations),
        };
        let params_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            gl_params,
        )?;

        let descriptor_set = PersistentDescriptorSet::start(self.layout.clone())
            .add_buffer(image_buffer.clone())?
            .add_buffer(params_buffer)?
            .build()?;

        // Create the commands to render the image and copy it to the buffer.
        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )?;
        let task_dimensions = [
            (params.image_size[0] + LOCAL_WORKGROUP_SIZES[0] - 1) / LOCAL_WORKGROUP_SIZES[0],
            (params.image_size[1] + LOCAL_WORKGROUP_SIZES[1] - 1) / LOCAL_WORKGROUP_SIZES[1],
            1,
        ];
        command_buffer
            .fill_buffer(image_buffer.clone(), 0)?
            .dispatch(task_dimensions, self.pipeline.clone(), descriptor_set, ())?;
        let command_buffer = command_buffer.build()?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        // Convert the buffer into an `ImageBuffer`.
        let buffer_content = image_buffer.read()?;
        debug_assert!(buffer_content.len() * 4 >= pixel_count);
        let buffer_content = unsafe {
            // SAFETY: Buffer length is correct by construction, and `[u8]` doesn't require
            // any special alignment.
            slice::from_raw_parts(buffer_content.as_ptr() as *const u8, pixel_count)
        };

        Ok(ImageBuffer::from_vec(
            params.image_size[0],
            params.image_size[1],
            buffer_content.to_vec(),
        )
        .unwrap())
    }
}
