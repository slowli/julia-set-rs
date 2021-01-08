#version 450
#define MAX_ITERATIONS 255

precision highp float;

vec2 complex_arg(vec2 a) {
    return vec2(atan(a.y, a.x), 0);
}

vec2 complex_sqrt(vec2 a) {
    float r = length(a);
    float im = sqrt((r - a.x) * 0.5);
    return vec2(sqrt(0.5 * (r + a.x)), (a.y >= 0.0) ? im : -im);
}

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 complex_div(vec2 a, vec2 b) {
    float denom = b.x * b.x + b.y * b.y;
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

vec2 complex_exp(vec2 a) {
    return vec2(cos(a.y), sin(a.y)) * exp(a.x);
}

vec2 complex_log(vec2 a) {
    return vec2(log(length(a)), atan(a.y, a.x));
}

vec2 complex_pow(vec2 a, vec2 b) {
    float arg = b.x * atan(a.y, a.x);
    float magn = pow(length(a), b.x);
    return vec2(cos(arg), sin(arg)) * magn;
}

vec2 complex_sinh(vec2 a) {
    float e = exp(a.x);
    float invE = 1.0 / e;
    return vec2((e - invE) * cos(a.y) * 0.5, (e + invE) * sin(a.y) * 0.5);
}

vec2 complex_cosh(vec2 a) {
    float e = exp(a.x);
    float invE = 1.0 / e;
    return vec2((e + invE) * cos(a.y) * 0.5, (e - invE) * sin(a.y) * 0.5);
}

vec2 complex_tanh(vec2 a) {
    float e = exp(2.0 * a.x);
    float invE = 1.0 / e;
    float denom = (e + invE) * 0.5 + cos(2.0 * a.y);
    return vec2((e - invE) * 0.5, sin(2.0 * a.y)) / denom;
}

float _r_acosh(float x) {
    return log(x) + log(1.0 + sqrt(1.0 - 1.0 / (x * x)));
}

vec2 complex_asinh(vec2 a) {
    float s = length(vec2(a.x, 1.0 + a.y));
    float t = length(vec2(a.x, 1.0 - a.y));
    float re = _r_acosh((s + t) * 0.5);
    return vec2((a.x >= 0.0) ? re : -re, asin(2.0 * a.y / (s + t)));
}

vec2 complex_acosh(vec2 a) {
    float p = length(vec2(1.0 + a.x, a.y));
    float q = length(vec2(1.0 - a.x, a.y));
    float im = acos(2.0 * a.x / (p + q));
    return vec2(_r_acosh((p + q) * 0.5), (a.y >= 0.0) ? im : -im);
}

vec2 complex_atanh(vec2 a) {
    float p = (1.0 + a.x) * (1.0 + a.x) + a.y * a.y;
    float q = (1.0 - a.x) * (1.0 - a.x) + a.y * a.y;
    return vec2(
        0.25 * (log(p) - log(q)),
        0.5 * atan(2.0 * a.y, 1.0 - a.x * a.x - a.y * a.y)
    );
}

vec2 compute(vec2 z) {
    COMPUTE
}

// **NB.** Local sizes must correspond to ones declared in the `vulkan` module.
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) writeonly buffer Output {
    uint[] data;
} output_buffer;

layout(set = 0, binding = 1) uniform Params {
    vec2 view_center;
    vec2 view_size;
    uvec2 image_size;
    float inf_distance_sq;
    uint max_iterations;
} params;

void main() {
    uvec2 image_size = params.image_size;
    if (gl_GlobalInvocationID.x >= image_size.x || gl_GlobalInvocationID.y >= image_size.y) {
        // We're out of image bounds, which happens if the image size is not divisible
        // by the local workgroup size.
        return;
    }

    // Map coordinates to [0, 1].
    vec2 pixel_pos = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(image_size);
    // Map coordinates to [-0.5, 0.5], flip the imaginary coordinate and transform
    // per view bounds.
    vec2 z = (pixel_pos - vec2(0.5)) * vec2(1 , -1) * params.view_size + params.view_center;

    uint iter = params.max_iterations;
    for (uint i = 0; i < MAX_ITERATIONS; i++) {
        z = compute(z);

        float z_sq_magnitude = z.x * z.x + z.y * z.y;
        if (
            i >= params.max_iterations ||
            isnan(z_sq_magnitude) ||
            isinf(z_sq_magnitude) ||
            z_sq_magnitude > params.inf_distance_sq
        ) {
            iter = i;
            break;
        }
    }

    float color = float(iter) / params.max_iterations;

    uint output_index = image_size.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x;
    uint mask = uint(round(color * 255));
    switch (output_index % 4) {
        case 1: mask <<= 8; break;
        case 2: mask <<= 16; break;
        case 3: mask <<= 24; break;
    }
    output_index /= 4;

    atomicOr(output_buffer.data[output_index], mask);
}
