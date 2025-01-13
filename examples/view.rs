#![allow(irrefutable_let_patterns)]

use blade_graphics as gpu;
use std::{env, fs, mem, slice};

mod spz {
    pub const MAGIC: u32 = 0x5053474e;
    pub const SCALE_LOG_SCALE: f32 = 16.0;
    pub const SCALE_LOG_OFFSET: f32 = -10.0;
    pub const ROT_SCALE: f32 = 1.0 / 127.5;

    #[repr(C)]
    #[derive(Default, Debug)]
    pub struct Header {
        pub magic: u32,
        pub version: u32,
        pub num_points: u32,
        pub sh_degree: u8,
        pub fractional_bits: u8,
        pub flags: u8,
        reserverd: u8,
    }
}

struct GaussianGpu {
    color: u32,
}

struct Icosahedron {
    vertices: [[f32; 3]; 12],
    triangles: [[u16; 3]; 20],
}

impl Icosahedron {
    fn new() -> Self {
        // http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
        let t0 = (1.0 + 5.0f32.sqrt()) / 2.0;
        let norm = (1.0 + t0 * t0).sqrt();
        let t = t0 / norm;
        let s = 1.0 / norm;
        Self {
            vertices: [
                [-s, t, 0.0],
                [s, t, 0.0],
                [-s, -t, 0.0],
                [s, -t, 0.0],
                [0.0, -s, t],
                [0.0, s, t],
                [0.0, -s, -t],
                [0.0, s, -t],
                [t, 0.0, -s],
                [t, 0.0, s],
                [-t, 0.0, -s],
                [-t, 0.0, s],
            ],
            triangles: [
                // 5 faces around point 0
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                // 5 adjacent faces
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                // 5 faces around point 3
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                // 5 adjacent faces
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
        }
    }
}

struct PointCloud {
    mesh_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    gauss_buf: gpu::Buffer,
    blas: gpu::AccelerationStructure,
    tlas: gpu::AccelerationStructure,
}

impl PointCloud {
    fn load(file_path: &str, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) -> Self {
        use std::io::Read as _;

        assert!(file_path.ends_with(".spz"));
        let spz_file = fs::File::open(file_path).unwrap();
        let mut gsz = flate2::read::GzDecoder::new(spz_file);
        let mut header = spz::Header::default();
        gsz.read_exact(unsafe {
            slice::from_raw_parts_mut(
                &mut header as *mut _ as *mut u8,
                mem::size_of::<spz::Header>(),
            )
        })
        .unwrap();
        log::info!("SPZ header: {:?}", header);
        assert_eq!(header.version, 2);
        assert_eq!(header.magic, spz::MAGIC);

        let count = header.num_points as usize;
        let mut instances = vec![gpu::AccelerationStructureInstance::default(); count];
        let mut scratch = Vec::<u8>::new();

        let gauss_total_size = (count * mem::size_of::<GaussianGpu>()) as u64;
        let gauss_buf = context.create_buffer(gpu::BufferDesc {
            name: "gauss-blobs",
            size: gauss_total_size,
            memory: gpu::Memory::Device,
        });
        let gauss_scratch = context.create_buffer(gpu::BufferDesc {
            name: "gauss-upload",
            size: gauss_total_size,
            memory: gpu::Memory::Upload,
        });

        let geometry = Icosahedron::new();
        let vertex_data_size = (geometry.vertices.len() * mem::size_of::<[f32; 3]>()) as u64;
        let index_data_size = (geometry.triangles.len() * mem::size_of::<[u16; 3]>()) as u64;
        let mesh_buf = context.create_buffer(gpu::BufferDesc {
            name: "gauss-mesh",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Device,
        });
        let meshes = [gpu::AccelerationStructureMesh {
            vertex_data: mesh_buf.at(0),
            vertex_format: gpu::VertexFormat::F32Vec3,
            vertex_stride: mem::size_of::<[f32; 3]>() as u32,
            vertex_count: geometry.vertices.len() as u32,
            index_data: mesh_buf.at(vertex_data_size),
            index_type: Some(gpu::IndexType::U16),
            triangle_count: geometry.triangles.len() as u32,
            transform_data: gpu::Buffer::default().at(0),
            is_opaque: false,
        }];
        let blas_sizes = context.get_bottom_level_acceleration_structure_sizes(&meshes);
        let blas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "blas",
            ty: gpu::AccelerationStructureType::BottomLevel,
            size: blas_sizes.data,
        });

        // positions
        let mut positions = vec![0u8; count * 3 * 24 / 8];
        gsz.read_exact(positions.as_mut_slice()).unwrap();
        // scales
        let mut scales = vec![0u8; count * 3];
        gsz.read_exact(scales.as_mut_slice()).unwrap();
        // rotations
        scratch.resize(count * 3, 0);
        gsz.read_exact(scratch.as_mut_slice()).unwrap();
        let pos_divisor = 1.0 / (1 << header.fractional_bits) as f32;
        for ((instance, p3), (r3, s3)) in instances
            .iter_mut()
            .zip(positions.chunks(3 * 24 / 8))
            .zip(scratch.chunks(3).zip(scales.chunks(3)))
        {
            let mut p_c = [0.0; 3];
            for (p_c1, p) in p_c.iter_mut().zip(p3.chunks(24 / 8)) {
                let pos_u = u32::from_le_bytes([p[0], p[1], p[2], 0]);
                *p_c1 = pos_u as f32 * pos_divisor;
            }
            let r = glam::Vec3::new(r3[0] as f32, r3[1] as f32, r3[2] as f32) * spz::ROT_SCALE;
            let q = glam::Quat::from_xyzw(r.x, r.y, r.z, (1.0 - r.dot(r)).sqrt());
            let s = glam::Vec3::new(s3[0] as f32, s3[1] as f32, s3[2] as f32)
                * spz::SCALE_LOG_SCALE
                + spz::SCALE_LOG_OFFSET;
            let m = glam::Mat3::from_quat(q) * glam::Mat3::from_diagonal(s);
            let col_major = mint::ColumnMatrix3x4 {
                x: m.x_axis.into(),
                y: m.y_axis.into(),
                z: m.z_axis.into(),
                w: p_c.into(),
            };
            instance.transform = col_major.into();
        }

        // alphas
        let mut alphas = vec![0u8; count];
        gsz.read_exact(alphas.as_mut_slice()).unwrap();
        // colors
        scratch.resize(count * 3, 0);
        gsz.read_exact(scratch.as_mut_slice()).unwrap();
        {
            let gaussians = unsafe {
                slice::from_raw_parts_mut(gauss_scratch.data() as *mut GaussianGpu, count)
            };
            for (gaussian, (c3, alpha)) in gaussians.iter_mut().zip(scratch.chunks(3).zip(alphas)) {
                gaussian.color = u32::from_le_bytes([c3[0], c3[1], c3[2], alpha]);
            }
        }

        // Build TLAS
        let tlas_sizes = context.get_top_level_acceleration_structure_sizes(count as u32);
        let instance_buf =
            context.create_acceleration_structure_instance_buffer(&instances, &[blas]);
        let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
            name: "TLAS",
            ty: gpu::AccelerationStructureType::TopLevel,
            size: tlas_sizes.data,
        });

        let tlas_scratch_offset =
            (blas_sizes.scratch | (gpu::limits::ACCELERATION_STRUCTURE_SCRATCH_ALIGNMENT - 1)) + 1;
        let scratch_buf = context.create_buffer(gpu::BufferDesc {
            name: "scratch",
            size: tlas_scratch_offset + tlas_sizes.scratch,
            memory: gpu::Memory::Device,
        });

        // Encode init operations
        encoder.start();
        if let mut pass = encoder.transfer("init") {
            pass.copy_buffer_to_buffer(gauss_scratch.at(0), gauss_buf.at(0), gauss_total_size);
        }
        if let mut pass = encoder.acceleration_structure("bottom") {
            pass.build_bottom_level(blas, &meshes, scratch_buf.at(0));
        }
        if let mut pass = encoder.acceleration_structure("top") {
            pass.build_top_level(
                tlas,
                &[blas],
                count as u32,
                instance_buf.at(0),
                scratch_buf.at(tlas_scratch_offset),
            );
        }
        let sync_point = context.submit(encoder);
        context.wait_for(&sync_point, !0);

        context.destroy_buffer(gauss_scratch);
        context.destroy_buffer(scratch_buf);

        Self {
            mesh_buf,
            instance_buf,
            gauss_buf,
            blas,
            tlas,
        }
    }

    fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.mesh_buf);
        context.destroy_buffer(self.gauss_buf);
        context.destroy_buffer(self.instance_buf);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_acceleration_structure(self.tlas);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Parameters {
    cam_position: [f32; 3],
    depth: f32,
    cam_orientation: [f32; 4],
    fov: [f32; 2],
    pad: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    g_parameters: Parameters,
    g_acc_struct: gpu::AccelerationStructure,
}

fn main() {
    env_logger::init();
    log::info!("Initializing");
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: !cfg!(debug_assertions),
            ..Default::default()
        })
    }
    .unwrap();
    let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "init",
        buffer_count: 1,
    });

    let arg_name = env::args()
        .nth(1)
        .expect("Need a path to .spz as an argument");
    let mut point_cloud = PointCloud::load(&arg_name, &context, &mut command_encoder);

    let extent = gpu::Extent {
        width: 100,
        height: 100,
        depth: 1,
    };
    let target_format = gpu::TextureFormat::Rgba8Unorm;
    let texture = context.create_texture(gpu::TextureDesc {
        name: "target",
        format: target_format,
        size: extent,
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: gpu::TextureDimension::D2,
        usage: gpu::TextureUsage::TARGET,
    });
    let texture_view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "target",
            format: target_format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );

    let source = std::fs::read_to_string("examples/shader.wgsl").unwrap();
    let shader = context.create_shader(gpu::ShaderDesc { source: &source });
    let draw_layout = <DrawData as gpu::ShaderData>::layout();
    let draw_pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
        name: "main",
        data_layouts: &[&draw_layout],
        primitive: gpu::PrimitiveState {
            topology: gpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        vertex: shader.at("draw_vs"),
        vertex_fetches: &[],
        fragment: Some(shader.at("draw_fs")),
        color_targets: &[target_format.into()],
        depth_stencil: None,
        multisample_state: Default::default(),
    });

    command_encoder.start();
    if let mut pass = command_encoder.render(
        "main",
        gpu::RenderTargetSet {
            colors: &[gpu::RenderTarget {
                view: texture_view,
                init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                finish_op: gpu::FinishOp::Store,
            }],
            depth_stencil: None,
        },
    ) {
        let mut pen = pass.with(&draw_pipeline);
        pen.bind(
            0,
            &DrawData {
                g_parameters: Parameters {
                    cam_position: [0.0, 0.0, 0.0],
                    depth: 1000.0,
                    cam_orientation: [0.0, 0.0, 0.0, 1.0],
                    fov: [0.7, 0.7],
                    pad: [0; 2],
                },
                g_acc_struct: point_cloud.tlas,
            },
        );
        pen.draw(0, 3, 0, 1);
    }
    let sync_point = context.submit(&mut command_encoder);
    context.wait_for(&sync_point, !0);
    context.destroy_texture_view(texture_view);
    context.destroy_texture(texture);

    context.destroy_command_encoder(&mut command_encoder);
    point_cloud.deinit(&context);
}
