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
    color: vec3f,
    opacity: f32,
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
    let ndc = 4.0 * tc - 1.0;
    let local_dir = vec3f(ndc * tan(0.5 * g_camera.fov), 1.0);
    vo.clip_pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
    vo.direction = qrot(g_camera.cam_orientation, local_dir);
    return vo;
}

const BACKGROUND: vec3f = vec3f(0.0);

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
        t_start = intersection.t * 1.0001;

        let gs = g_data[intersection.instance_id];
        let basis = gs.opacity / g_params.min_opacity;

        let object_ray_pos = intersection.world_to_object * vec4f(ray_pos, 1.0);
        let object_ray_dir = normalize(intersection.world_to_object * vec4f(ray_dir, 0.0));
        let effective_t = -dot(object_ray_pos, object_ray_dir);

        let object_pos = object_ray_pos + effective_t * object_ray_dir;
        let alpha = gs.opacity * pow(basis, -dot(object_pos, object_pos));
        if (alpha > g_params.min_opacity) {
            //TODO: evaluate spherical harmonics here
            radiance += alpha * transmittance * gs.color;
            transmittance *= 1.0 - alpha;
        }
    }

    radiance += transmittance * BACKGROUND;
    return vec4f(radiance, 1.0);
}
