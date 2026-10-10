// Included as string constants in the native bridge. Model orchestration is Rust.
// W1 storage remains packed; activation quantization is [-7,7], group size 32.
static const char* pack_source = R"metal(
    uint g = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    if (g >= G) return;
    float a = x[g * 32 + lane];
    float maximum = simd_max(abs(a));
    int q = maximum == 0.0f ? 0 : int(clamp(round((a / maximum) * 7.0f), -7.0f, 7.0f));
    uint m = uint(abs(q));
    // Each lane contributes a distinct bit; unsigned sum equals bitwise OR.
    uint m0 = simd_sum((m & 1) != 0 ? (1u << lane) : 0u);
    uint m1 = simd_sum((m & 2) != 0 ? (1u << lane) : 0u);
    uint m2 = simd_sum((m & 4) != 0 ? (1u << lane) : 0u);
    uint sign = simd_sum(q < 0 ? (1u << lane) : 0u);
    int total = simd_sum(q);
    int negative = simd_sum(max(-q, 0));
    if (lane == 0) {
        planes[g * 4] = m0; planes[g * 4 + 1] = m1;
        planes[g * 4 + 2] = m2; planes[g * 4 + 3] = sign;
        scales[g] = maximum / 7.0f;
        stats[g * 2] = total; stats[g * 2 + 1] = negative;
    }
)metal";

static const char* project_source = R"metal(
    #pragma clang fp contract(off)
    uint row = thread_position_in_grid.x;
    if (row >= N) return;
    float total = 0.0f;
    for (uint wg = 0; wg < G / 4; ++wg) {
        float ws = weight_scales[wg * N + row];
        float wb = weight_biases[wg * N + row];
        for (uint j = 0; j < 4; ++j) {
            uint g = wg * 4 + j;
            uint r = weights[g * N + row] ^ planes[g * 4 + 3];
            int c = int(popcount(r & planes[g * 4]))
                + 2 * int(popcount(r & planes[g * 4 + 1]))
                + 4 * int(popcount(r & planes[g * 4 + 2]));
            int dot = c - stats[g * 2 + 1];
            total += (ws * float(dot) + wb * float(stats[g * 2])) * scales[g];
        }
    }
    y[row] = total;
)metal";

static const char* embed_source = R"metal(
    #pragma clang fp contract(off)
    uint d = thread_position_in_grid.x;
    if (d >= K) return;
    uint row = token[0];
    uint w = weights[(d / 32) * N + row];
    float bit = float((w >> (d % 32)) & 1);
    y[d] = bit * weight_scales[(d / 128) * N + row] + weight_biases[(d / 128) * N + row];
)metal";

static const char* rope_source = R"metal(
    #pragma clang fp contract(off)
    uint i = thread_position_in_grid.x;
    if (i >= NH * HD / 2) return;
    uint head = i / (HD / 2), d = i % (HD / 2);
    float angle = float(position[0]) * frequencies[d];
    float s = sin(angle) * attention_scale[0], c = cos(angle) * attention_scale[0];
    float a = x[head * HD + d], b = x[head * HD + d + HD / 2];
    y[head * HD + d] = a * c - b * s;
    y[head * HD + d + HD / 2] = a * s + b * c;
)metal";
