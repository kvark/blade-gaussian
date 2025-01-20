#![allow(irrefutable_let_patterns)]

mod point_cloud;
mod shape;

pub use point_cloud::PointCloud;
pub use shape::Icosahedron;

pub struct GaussianGpu {
    pub color: [f32; 4],
}
