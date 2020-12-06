#define MAX_ITERATIONS 100

struct __attribute__((packed)) Params {
    // View center coordinates.
    float2 view_center;
    // View sizes.
    float2 view_size;
    // Square of the infinity distance.
    float inf_distance_sq;
};

float2 complex_arg(float2 a) {
    return (float2)(atan2(a.y, a.x), 0);
}

float2 complex_sqrt(float2 a) {
    float r = length(a);
    float im = sqrt((r - a.x) * 0.5);
    return (float2)(sqrt(0.5 * (r + a.x)), (a.y >= 0.0) ? im : -im);
}

float2 complex_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 complex_div(float2 a, float2 b) {
    float denom = b.x * b.x + b.y * b.y;
    return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

float2 complex_exp(float2 a) {
    return (float2)(cos(a.y), sin(a.y)) * exp(a.x);
}

float2 complex_pow(float2 a, float2 b) {
    float arg = b.x * atan2(a.y, a.x);
    float magn = pow(length(a), b.x);
    return (float2)(cos(arg), sin(arg)) * magn;
}

float2 complex_sinh(float2 a) {
    float e = exp(a.x);
    float invE = 1.0 / e;
    return (float2)((e - invE) * cos(a.y) * 0.5, (e + invE) * sin(a.y) * 0.5);
}

float2 complex_cosh(float2 a) {
    float e = exp(a.x);
    float invE = 1.0 / e;
    return (float2)((e + invE) * cos(a.y) * 0.5, (e - invE) * sin(a.y) * 0.5);
}

float2 complex_tanh(float2 a) {
    float e = exp(2.0 * a.x);
    float invE = 1.0 / e;
    float denom = (e + invE) * 0.5 + cos(2.0 * a.y);
    return (float2)((e - invE) * 0.5, sin(2.0 * a.y)) / denom;
}

float _r_acosh(float x) {
    return log(x) + log(1.0 + sqrt(1.0 - 1.0 / (x * x)));
}

float2 complex_asinh(float2 a) {
    float s = length((float2)(a.x, 1.0 + a.y));
    float t = length((float2)(a.x, 1.0 - a.y));
    float re = _r_acosh((s + t) * 0.5);
    return (float2)((a.x >= 0.0) ? re : -re, asin(2.0 * a.y / (s + t)));
}

float2 complex_acosh(float2 a) {
    float p = length((float2)(1.0 + a.x, a.y));
    float q = length((float2)(1.0 - a.x, a.y));
    float im = acos(2.0 * a.x / (p + q));
    return (float2)(_r_acosh((p + q) * 0.5), (a.y >= 0.0) ? im : -im);
}

float2 complex_atanh(float2 a) {
    float p = (1.0 + a.x) * (1.0 + a.x) + a.y * a.y;
    float q = (1.0 - a.x) * (1.0 - a.x) + a.y * a.y;
    return (float2)(
        0.25 * (log(p) - log(q)),
        0.5 * atan2(2.0 * a.y, 1.0 - a.x * a.x - a.y * a.y)
    );
}

float2 compute(float2 z) {
    COMPUTE(z)
}

__kernel void julia(
    __global uchar *output,
    struct Params params
) {
    // Retrieve image dimensions and coordinates.
    size_t w = get_num_groups(0) * get_local_size(0);
    size_t h = get_num_groups(1) * get_local_size(1);
    size_t pixel_x = get_global_id(0);
    size_t pixel_y = get_global_id(1);

    // Convert pixels to the real-valued coordinates.
    float2 z = (float2)(pixel_x + 0.5, pixel_y + 0.5) / (float2)(w, h) - (float2) 0.5;
    z *= (float2)(1, -1); // flip the imaginary coordinate
    z = z * params.view_size + params.view_center;

    uchar iter = MAX_ITERATIONS;
    __attribute__((opencl_unroll_hint))
    for (uchar i = 0; i < MAX_ITERATIONS; i++) {
        z = compute(z);
        if (z.x * z.x + z.y * z.y > params.inf_distance_sq) {
            iter = i;
            break;
        }
    }

    float color = (float) iter / MAX_ITERATIONS;
    color = smoothstep(0.0f, 1.0f, 1.0f - color);
    output[pixel_y * w + pixel_x] = round(color * 255);
}
