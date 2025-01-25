#![allow(irrefutable_let_patterns)]

pub mod io;
mod point_cloud;
mod shape;

pub use point_cloud::{InitParameters, PointCloud};
pub use shape::Icosahedron;

pub const SH_DEGREE: usize = 0;
pub const SH_COMPONENTS: usize = (1 + SH_DEGREE) * (1 + SH_DEGREE);

#[derive(Clone, Default)]
pub struct Gaussian {
    pub mean: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub opacity: f32,
    pub sh: [glam::Vec3; SH_COMPONENTS],
}

pub struct GaussianGpu {
    pub color: [f32; 4],
}
