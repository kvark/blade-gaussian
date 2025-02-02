struct Camera {
    position: vec3<f32>,
    depth: f32,
    orientation: vec4<f32>,
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
    mean: vec3f,
    pad1: f32,
    rotation: vec4f,
    scale: vec3f,
    pad2: f32,
}
var<storage> g_data: array<Gaussian>;

fn qmake(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    return vec4<f32>(axis * sin(angle), cos(angle));
}
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
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
    vo.direction = qrot(g_camera.orientation, local_dir);
    return vo;
}

fn check_intersection(intersection: RayIntersection, ray_pos: vec3f, ray_dir: vec3f) -> bool {
    let gs = g_data[intersection.instance_index];
    let object_ray_pos = intersection.world_to_object * vec4f(ray_pos, 1.0);
    let object_ray_dir = intersection.world_to_object * vec4f(ray_dir, 0.0);
    let effective_t = -dot(object_ray_pos, object_ray_dir) / dot(object_ray_dir, object_ray_dir);
    let object_pos = object_ray_pos + effective_t * object_ray_dir;
    return dot(object_pos, object_pos) <= 1.0;
}

struct Hit {
    t: f32,
    i: u32,
}

const hit_window: u32 = 5;
const BACKGROUND: vec3f = vec3f(0.0);

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    let ray_pos = g_camera.position;
    let ray_dir = normalize(vo.direction);
    var t_start = 0.0;
    var transmittance = 1.0;
    var radiance = vec3f(0.0);

    while (transmittance > g_params.min_transmittance) {
        var rq: ray_query;
        let ray_flags = RAY_FLAG_CULL_BACK_FACING | RAY_FLAG_FORCE_NO_OPAQUE;
        let desc = RayDesc(ray_flags, 0xFFu, t_start, g_camera.depth, ray_pos, ray_dir);
        rayQueryInitialize(&rq, g_acc_struct, desc);
        var hit_count = 0u;
        var hits: array<Hit, hit_window>;
        var in_progress = true;

        while (in_progress) {
            in_progress = rayQueryProceed(&rq);
            let intersection = rayQueryGetCandidateIntersection(&rq);
            if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
                continue;
            }
            if (!check_intersection(intersection, ray_pos, ray_dir)) {
                continue;
            }

            var hit = Hit(intersection.t, intersection.instance_index);
            for (var i = 0u; i < hit_count; i += 1u) {
                let other = hits[i];
                if (hit.t < other.t) {
                    hits[i] = hit;
                    hit = other;
                }
            }
            if (hit_count < hit_window) {
                hits[hit_count] = hit;
                hit_count += 1;
            }
            if (hit_count == hit_window && intersection.t >= hits[hit_window - 1u].t) {
                rayQueryConfirmIntersection(&rq);
            }
        }

        for (var i = 0u; i < hit_count; i += 1u) {
            let hit = hits[i];
            let gs = g_data[hit.i];

            let g_origin = qrot(qinv(gs.rotation), ray_pos - gs.mean) / gs.scale;
            let g_dir = qrot(qinv(gs.rotation), ray_dir) / gs.scale;
            let effective_t = -dot(g_origin, g_dir) / dot(g_dir, g_dir);
            let g_pos = g_origin + effective_t * g_dir;
            let alpha = gs.opacity * exp(-0.5 * dot(g_pos, g_pos));

            //TODO: evaluate spherical harmonics here
            radiance += alpha * transmittance * gs.color;
            transmittance *= 1.0 - alpha;
            t_start = hit.t;
        }

        t_start *= 1.00001; // avoid hitting the same primitive
        if (hit_count < hit_window) {
            break;
        }
    }

    radiance += transmittance * BACKGROUND;
    return vec4f(radiance, 1.0);
}
