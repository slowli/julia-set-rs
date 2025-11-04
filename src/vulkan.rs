//! GLSL / Vulkan backend for Julia sets.

use std::{slice, sync::Arc};

use anyhow::{Context as _, anyhow};
use shaderc::{CompilationArtifact, CompileOptions, OptimizationLevel, ShaderKind};
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

use crate::{Backend, Function, ImageBuffer, Params, Render, compiler::Compiler};

const PROGRAM: &str = include_str!(concat!(env!("OUT_DIR"), "/program.glsl"));

const LOCAL_WORKGROUP_SIZES: [u32; 2] = [16, 16];

fn compile_shader(function: &str) -> shaderc::Result<CompilationArtifact> {
    let compiler = shaderc::Compiler::new()?;
    let mut options = CompileOptions::new()?;
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

#[derive(Debug, Clone, Copy, BufferContents)]
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
    pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl VulkanProgram {
    fn new(compiled_function: &str) -> anyhow::Result<Self> {
        let library = VulkanLibrary::new()?;
        let create_info = InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..InstanceCreateInfo::default()
        };
        let instance = Instance::new(library, create_info)?;

        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };
        let (device, queue_family_index) =
            Self::select_physical_device(&instance, &device_extensions)?;
        let (device, mut queues) = Device::new(
            device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..QueueCreateInfo::default()
                }],
                ..DeviceCreateInfo::default()
            },
        )?;
        let queue = queues
            .next()
            .with_context(|| format!("cannot initialize compute queue on device {device:?}"))?;

        let shader = compile_shader(compiled_function)?;
        let create_info = ShaderModuleCreateInfo::new(shader.as_binary());
        let shader = unsafe { ShaderModule::new(device.clone(), create_info)? };
        let entry_point = shader
            .entry_point("main")
            .context("cannot find entry point `main` in Julia set compute shader")?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .context("cannot create compute pipeline layout")?,
        )
        .context("failed validating pipeline layout")?;

        let create_info = ComputePipelineCreateInfo::stage_layout(stage, layout);
        let pipeline = ComputePipeline::new(device.clone(), None, create_info)?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        Ok(Self {
            device,
            queue,
            pipeline,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    fn select_physical_device(
        instance: &Arc<Instance>,
        device_extensions: &DeviceExtensions,
    ) -> anyhow::Result<(Arc<PhysicalDevice>, u32)> {
        let devices = instance.enumerate_physical_devices()?;
        let devices =
            devices.filter(|device| device.supported_extensions().contains(device_extensions));
        let devices = devices.filter_map(|device| {
            device
                .queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|idx| (device, idx as u32))
        });

        let device = devices
            .min_by_key(|(device, _)| Self::device_type_priority(device.properties().device_type));
        device.ok_or_else(|| anyhow!("failed selecting physical device with compute queue"))
    }

    fn device_type_priority(ty: PhysicalDeviceType) -> usize {
        match ty {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        }
    }
}

impl Render for VulkanProgram {
    type Error = anyhow::Error;

    #[allow(clippy::cast_possible_truncation)]
    fn render(&self, params: &Params) -> anyhow::Result<ImageBuffer> {
        // Bind uniforms: the output image buffer and the rendering params.
        let pixel_count = (params.image_size[0] * params.image_size[1]) as usize;
        let image_buffer = Buffer::new_slice::<u32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..AllocationCreateInfo::default()
            },
            pixel_count as u64,
        )?;

        let gl_params = VulkanParams {
            view_center: params.view_center,
            view_size: [params.view_width(), params.view_height],
            image_size: params.image_size,
            inf_distance_sq: params.inf_distance * params.inf_distance,
            max_iterations: u32::from(params.max_iterations),
        };

        let layout = self.pipeline.layout();
        let layout = &layout.set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, image_buffer.clone())],
            [],
        )?;

        // Create the commands to render the image and copy it to the buffer.
        let task_dimensions = [
            params.image_size[0].div_ceil(LOCAL_WORKGROUP_SIZES[0]),
            params.image_size[1].div_ceil(LOCAL_WORKGROUP_SIZES[1]),
            1,
        ];
        let layout = self.pipeline.layout();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .context("failed binding compute pipeline to command buffer")?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                descriptor_set,
            )
            .context("failed binding descriptor sets to command buffer")?
            .fill_buffer(image_buffer.clone(), 0)?
            .push_constants(layout.clone(), 0, gl_params)
            .context("failed pushing constants to command buffer")?;
        unsafe {
            builder.dispatch(task_dimensions)?;
        }
        let command_buffer = builder.build()?;

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
            slice::from_raw_parts(buffer_content.as_ptr().cast::<u8>(), pixel_count)
        };

        Ok(ImageBuffer::from_vec(
            params.image_size[0],
            params.image_size[1],
            buffer_content.to_vec(),
        )
        .unwrap())
    }
}
