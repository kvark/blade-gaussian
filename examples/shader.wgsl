struct Camera {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
    pad: vec2<u32>,
}

var<uniform> g_camera: Camera;

struct Parameters {
    min_opacity: f32,
    min_transmittance: f32,
}
var<uniform> g_params: Parameters;
var g_acc_struct: acceleration_structure;

struct Gaussian {
    color: vec4f,
}
var<storage> g_data: array<Gaussian>;

fn qmake(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(angle), cos(angle));
}
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) direction: vec3<f32>,
}

@vertex
fn draw_vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo = VertexOutput();
    let tc = vec2f(vec2u(vi) & vec2u(1u, 2u)) * vec2f(1.0, 0.5);
    let ndc = 2.0 * tc - 1.0;
    let local_dir = vec3f(ndc * tan(g_camera.fov), 1.0);
    vo.clip_pos = vec4f(4.0 * tc.x - 1.0, 1.0 - 4.0 * tc.y, 0.0, 1.0);
    vo.direction = qrot(g_camera.cam_orientation, local_dir);
    return vo;
}

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    var rq: ray_query;
    let ray_pos = g_camera.cam_position;
    let ray_dir = normalize(vo.direction);
    var t_start = 0.0;
    var transmittance = 1.0;
    var radiance = vec3f(0.0);

    while (transmittance > g_params.min_transmittance) {
        let desc = RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING, 0xFFu, t_start, g_camera.depth, ray_pos, ray_dir);
        rayQueryInitialize(&rq, g_acc_struct, desc);
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            break;
        }
        t_start = intersection.t * 1.001;

        let gs = g_data[intersection.instance_id];
        let world_pos = ray_pos + intersection.t * ray_dir;
        let local_pos = intersection.world_to_object * vec4f(world_pos, 1.0);
        let alpha = gs.color.a * exp(-0.5 * dot(local_pos, local_pos));
        if (alpha > g_params.min_opacity) {
            //TODO: evaluate spherical harmonics here
            radiance += alpha * transmittance * (gs.color.xyz + 0.5);
            transmittance *= 1.0 - alpha;
        }
    }

    return vec4f(radiance, 0.0);
}
