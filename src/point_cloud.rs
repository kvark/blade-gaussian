use blade_graphics as gpu;

use std::{fs, mem, ptr, slice};

mod spz {
    pub const MAGIC: u32 = 0x5053474e;
    pub const SCALE_LOG_SCALE: f32 = 1.0 / 16.0;
    pub const SCALE_LOG_OFFSET: f32 = -10.0;
    pub const ROT_SCALE: f32 = 1.0 / 127.5;
    pub const COLOR_SCALE: f32 = 1.0 / 0.15;

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

fn inv_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}
fn unpack_color(raw: u8) -> f32 {
    spz::COLOR_SCALE * (raw as f32 / 255.0 - 0.5)
}

pub struct PointCloud {
    mesh_buf: gpu::Buffer,
    instance_buf: gpu::Buffer,
    pub gauss_buf: gpu::Buffer,
    blas: gpu::AccelerationStructure,
    pub tlas: gpu::AccelerationStructure,
}

impl PointCloud {
    pub fn load(
        file_path: &str,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Self {
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

        let gauss_total_size = (count * mem::size_of::<super::GaussianGpu>()) as u64;
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

        let inner_radius = 1.0; //TODO
        let geometry = super::Icosahedron::new(inner_radius);
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
                let sign = if p[2] & 0x80 != 0 { -1.0 } else { 1.0 };
                *p_c1 = sign * pos_u as f32 * pos_divisor;
            }
            let r = glam::Vec3::new(r3[0] as i8 as f32, r3[1] as i8 as f32, r3[2] as i8 as f32)
                * spz::ROT_SCALE;
            let q = glam::Quat::from_xyzw(r.x, r.y, r.z, (1.0 - r.dot(r)).max(0.0).sqrt());
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
                slice::from_raw_parts_mut(gauss_scratch.data() as *mut super::GaussianGpu, count)
            };
            for (gaussian, (c3, raw_alpha)) in
                gaussians.iter_mut().zip(scratch.chunks(3).zip(alphas))
            {
                let alpha = inv_sigmoid(raw_alpha as f32 / 255.0);
                gaussian.color = [
                    unpack_color(c3[0]),
                    unpack_color(c3[1]),
                    unpack_color(c3[2]),
                    alpha,
                ];
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

        let mesh_stage = context.create_buffer(gpu::BufferDesc {
            name: "gauss-mesh-stage",
            size: vertex_data_size + index_data_size,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(
                geometry.vertices.as_ptr(),
                mesh_stage.data() as *mut [f32; 3],
                geometry.vertices.len(),
            );
            ptr::copy_nonoverlapping(
                geometry.triangles.as_ptr(),
                mesh_stage.data().add(vertex_data_size as usize) as *mut [u16; 3],
                geometry.triangles.len(),
            );
        }

        // Encode init operations
        encoder.start();
        if let mut pass = encoder.transfer("init") {
            pass.copy_buffer_to_buffer(
                mesh_stage.at(0),
                mesh_buf.at(0),
                vertex_data_size + index_data_size,
            );
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
        context.destroy_buffer(mesh_stage);

        Self {
            mesh_buf,
            instance_buf,
            gauss_buf,
            blas,
            tlas,
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_buffer(self.mesh_buf);
        context.destroy_buffer(self.gauss_buf);
        context.destroy_buffer(self.instance_buf);
        context.destroy_acceleration_structure(self.blas);
        context.destroy_acceleration_structure(self.tlas);
    }
}
