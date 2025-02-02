#![allow(irrefutable_let_patterns)]

pub mod io;
mod point_cloud;
mod shape;

pub use point_cloud::{InitParameters, PointCloud};
pub use shape::Icosahedron;

pub const fn get_sh_component_count(degree: usize) -> usize {
    (1 + degree) * (1 + degree)
}
pub const fn get_sh_degree(count: usize) -> usize {
    if count <= 1 {
        0
    } else if count <= 4 {
        1
    } else if count <= 9 {
        2
    } else {
        3
    }
}

pub const MAX_SH_DEGREE: usize = 0;
pub const MAX_SH_COMPONENTS: usize = get_sh_component_count(MAX_SH_DEGREE);

#[derive(Clone, Default)]
pub struct Gaussian {
    pub mean: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub opacity: f32,
    pub shc: [glam::Vec3; MAX_SH_COMPONENTS],
}

pub struct Model {
    pub gaussians: Vec<Gaussian>,
    pub max_sh_degree: usize,
}

#[repr(C)]
pub struct GaussianGpu {
    pub color: [f32; 3],
    pub opacity: f32,
    pub mean: [f32; 3],
    pub pad1: f32,
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub pad2: f32,
}
