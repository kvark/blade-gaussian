struct Parameters {
    cam_position: vec3<f32>,
    depth: f32,
    cam_orientation: vec4<f32>,
    fov: vec2<f32>,
    pad: vec2<u32>,
};

var<uniform> g_parameters: Parameters;
var g_acc_struct: acceleration_structure;

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
    let local_dir = vec3f(ndc * tan(g_parameters.fov), 1.0);
    vo.clip_pos = vec4f(4.0 * tc.x - 1.0, 1.0 - 4.0 * tc.y, 0.0, 1.0);
    vo.direction = qrot(g_parameters.cam_orientation, local_dir);
    return vo;
}

@fragment
fn draw_fs(vo: VertexOutput) -> @location(0) vec4<f32> {
    var rq: ray_query;
    let ray_pos = g_parameters.cam_position;
    let ray_dir = normalize(vo.direction);
    var t_start = 0.0;
    var color = vec3f(0.0);
    loop {
        let desc = RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING, 0xFFu, t_start + 0.1, g_parameters.depth, ray_pos, ray_dir);
        rayQueryInitialize(&rq, g_acc_struct, desc);
        rayQueryProceed(&rq);
        let intersection = rayQueryGetCommittedIntersection(&rq);
        if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) {
            break;
        }
        t_start = intersection.t;
    }

    color = vec3f(t_start);
    return vec4f(color, 0.0);
}
