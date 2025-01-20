#![allow(irrefutable_let_patterns)]

use blade_gaussian as gauss;
use blade_graphics as gpu;
use std::env;

const MAX_FLY_SPEED: f32 = 1000000.0;

#[derive(Default)]
pub struct ControlledCamera {
    pub position: glam::Vec3,
    pub orientation: glam::Quat,
    pub fov_y: f32,
    pub depth: f32,
    pub fly_speed: f32,
}

impl ControlledCamera {
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.orientation, self.position).inverse()
    }

    pub fn get_projection_matrix(&self, aspect: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_y, aspect, 1.0, self.depth)
    }

    pub fn move_by(&mut self, offset: glam::Vec3) {
        self.position += self.orientation * offset;
    }

    pub fn rotate_z_by(&mut self, angle: f32) {
        self.orientation *= glam::Quat::from_rotation_z(angle);
    }

    pub fn on_key(&mut self, code: winit::keyboard::KeyCode, delta: f32) -> bool {
        use winit::keyboard::KeyCode as Kc;

        let move_offset = self.fly_speed * delta;
        let rotate_offset_z = 1000.0 * delta;
        match code {
            Kc::KeyW => {
                self.move_by(glam::Vec3::new(0.0, 0.0, -move_offset));
            }
            Kc::KeyS => {
                self.move_by(glam::Vec3::new(0.0, 0.0, move_offset));
            }
            Kc::KeyA => {
                self.move_by(glam::Vec3::new(-move_offset, 0.0, 0.0));
            }
            Kc::KeyD => {
                self.move_by(glam::Vec3::new(move_offset, 0.0, 0.0));
            }
            Kc::KeyZ => {
                self.move_by(glam::Vec3::new(0.0, -move_offset, 0.0));
            }
            Kc::KeyX => {
                self.move_by(glam::Vec3::new(0.0, move_offset, 0.0));
            }
            Kc::KeyQ => {
                self.rotate_z_by(rotate_offset_z);
            }
            Kc::KeyE => {
                self.rotate_z_by(-rotate_offset_z);
            }
            _ => return false,
        }

        true
    }

    pub fn on_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
        let shift = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, lines) => lines,
            winit::event::MouseScrollDelta::PixelDelta(position) => position.y as f32,
        };
        self.fly_speed = (self.fly_speed * shift.exp()).clamp(1.0, MAX_FLY_SPEED);
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

struct Example {
    camera: ControlledCamera,
    draw_pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    prev_sync_point: Option<gpu::SyncPoint>,
    window_size: winit::dpi::PhysicalSize<u32>,
    point_cloud: gauss::PointCloud,
    surface: gpu::Surface,
    context: gpu::Context,
}

impl Example {
    fn make_surface_config(size: winit::dpi::PhysicalSize<u32>) -> gpu::SurfaceConfig {
        log::info!("Window size: {:?}", size);
        gpu::SurfaceConfig {
            size: gpu::Extent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            ..Default::default()
        }
    }

    fn init(window: &winit::window::Window) -> Self {
        let context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                timing: false,
                capture: false,
                overlay: true,
                device_id: 0,
            })
            .unwrap()
        };
        println!("{:?}", context.device_information());
        let window_size = window.inner_size();

        let surface = context
            .create_surface_configured(window, Self::make_surface_config(window_size))
            .unwrap();
        let info = surface.info();

        let shader = {
            let source = std::fs::read_to_string("examples/shader.wgsl").unwrap();
            context.create_shader(gpu::ShaderDesc { source: &source })
        };
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
            color_targets: &[info.format.into()],
            depth_stencil: None,
            multisample_state: Default::default(),
        });

        let mut command_encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        let arg_name = env::args()
            .nth(1)
            .expect("Need a path to .spz as an argument");
        let point_cloud = gauss::PointCloud::load(&arg_name, &context, &mut command_encoder);

        Self {
            camera: ControlledCamera {
                depth: 10000.0,
                fly_speed: 1.0,
                ..Default::default()
            },
            draw_pipeline,
            command_encoder,
            prev_sync_point: None,
            window_size,
            point_cloud,
            surface,
            context,
        }
    }

    fn deinit(&mut self) {
        self.wait_for_gpu();
        self.context
            .destroy_render_pipeline(&mut self.draw_pipeline);
        self.context
            .destroy_command_encoder(&mut self.command_encoder);
        self.context.destroy_surface(&mut self.surface);
        self.point_cloud.deinit(&self.context);
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = size;
        let config = Self::make_surface_config(size);
        self.context.reconfigure_surface(&mut self.surface, config);
    }

    fn wait_for_gpu(&mut self) {
        if let Some(sp) = self.prev_sync_point.take() {
            self.context.wait_for(&sp, !0);
        }
    }

    fn render(&mut self) {
        if self.window_size == Default::default() {
            return;
        }
        let frame = self.surface.acquire_frame();
        let aspect = self.window_size.width as f32 / self.window_size.height as f32;

        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());

        if let mut pass = self.command_encoder.render(
            "main",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            let mut pen = pass.with(&self.draw_pipeline);
            pen.bind(
                0,
                &DrawData {
                    g_parameters: Parameters {
                        cam_position: self.camera.position.into(),
                        depth: self.camera.depth,
                        cam_orientation: self.camera.orientation.into(),
                        fov: [aspect * self.camera.fov_y, self.camera.fov_y],
                        pad: [0; 2],
                    },
                    g_acc_struct: self.point_cloud.tlas,
                },
            );
            pen.draw(0, 3, 0, 1);
        }
        self.command_encoder.present(frame);
        let sync_point = self.context.submit(&mut self.command_encoder);

        self.wait_for_gpu();
        self.prev_sync_point = Some(sync_point);
    }
}
fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let window_attributes =
        winit::window::Window::default_attributes().with_title("blade-gaussian-viewer");
    let window = event_loop.create_window(window_attributes).unwrap();

    let mut example = Example::init(&window);

    event_loop
        .run(|event, target| {
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);
            match event {
                winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }
                winit::event::Event::WindowEvent { event, .. } => match event {
                    winit::event::WindowEvent::Resized(size) => {
                        example.resize(size);
                    }
                    winit::event::WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        if key_code == winit::keyboard::KeyCode::Escape {
                            target.exit();
                        }
                        example.camera.on_key(key_code, 1.0);
                    }
                    winit::event::WindowEvent::MouseWheel { delta, .. } => {
                        example.camera.on_wheel(delta);
                    }
                    winit::event::WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        example.render();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();

    example.deinit();
}
