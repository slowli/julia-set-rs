#define MAX_ITERATIONS 100

struct __attribute__((packed)) Params {
    // View center coordinates.
    float2 view_center;
    // View sizes.
    float2 view_size;
    // Square of the infinity distance.
    float inf_distance_sq;
};

float2 complex_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
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
