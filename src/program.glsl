#version 450
#define MAX_ITERATIONS 100

precision highp float;

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 compute(vec2 z) {
    COMPUTE
}

layout(set = 0, binding = 0, r8) uniform writeonly image2D img;
layout(set = 0, binding = 1) uniform Params {
    vec2 view_center;
    vec2 view_size;
    float inf_distance_sq;
} params;

void main() {
    // Map coordinates to [0, 1].
    vec2 pixel_pos = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    // Map coordinates to [-0.5, 0.5], flip the imaginary coordinate and transform
    // per view bounds.
    vec2 z = (pixel_pos - vec2(0.5)) * vec2(1 , -1) * params.view_size + params.view_center;

    uint iter = MAX_ITERATIONS;
    for (uint i = 0; i < MAX_ITERATIONS; i++) {
        z = compute(z);
        if (z.x * z.x + z.y * z.y > params.inf_distance_sq) {
            iter = i;
            break;
        }
    }

    float color = float(iter) / MAX_ITERATIONS;
    color = smoothstep(0.0, 1.0, 1.0 - color);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(color));
}
