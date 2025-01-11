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

struct _GaussianGpu {
    alpha: f32,
    color: u32,
}

fn main() {
    use std::io::Read as _;

    env_logger::init();
    log::info!("Initializing");
    let context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            ..Default::default()
        })
    }
    .unwrap();

    let arg_name = env::args()
        .nth(1)
        .expect("Need a path to .spz as an argument");
    assert!(arg_name.ends_with(".spz"));
    let spz_file = fs::File::open(arg_name).unwrap();
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

    let vertex_buf = context.create_buffer(gpu::BufferDesc {
        name: "gauss",
        size: 16,
        memory: gpu::Memory::Device,
    });
    let meshes = [gpu::AccelerationStructureMesh {
        vertex_data: vertex_buf.at(0),
        vertex_format: gpu::VertexFormat::F32Vec3,
        vertex_stride: mem::size_of::<f32>() as u32 * 3,
        vertex_count: 1,
        index_data: gpu::Buffer::default().at(0),
        index_type: None,
        triangle_count: 0,
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
        let s = glam::Vec3::new(s3[0] as f32, s3[1] as f32, s3[2] as f32) * spz::SCALE_LOG_SCALE
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

    // Build TLAS
    let tlas_sizes = context.get_top_level_acceleration_structure_sizes(count as u32);
    let instance_buffer =
        context.create_acceleration_structure_instance_buffer(&instances, &[blas]);
    let tlas = context.create_acceleration_structure(gpu::AccelerationStructureDesc {
        name: "TLAS",
        ty: gpu::AccelerationStructureType::TopLevel,
        size: tlas_sizes.data,
    });

    // Cleanup
    context.destroy_buffer(vertex_buf);
    context.destroy_buffer(instance_buffer);
    context.destroy_acceleration_structure(blas);
    context.destroy_acceleration_structure(tlas);
}
