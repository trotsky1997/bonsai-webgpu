// Exception-safe C ABI over official MLX array operations. No model logic here.
#include <mlx/mlx.h>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include "w1a4.metal"
using namespace mlx::core;
static thread_local std::string last_error;
#define TRY try
#define FAIL_NULL catch (const std::exception& e) { last_error = e.what(); return nullptr; } catch (...) { last_error = "unknown MLX exception"; return nullptr; }
#define FAIL_INT catch (const std::exception& e) { last_error = e.what(); return -1; } catch (...) { last_error = "unknown MLX exception"; return -1; }
static Shape shape(const int* dims, int rank) { return Shape(dims, dims + rank); }
extern "C" {
const char* bn_error() { return last_error.c_str(); }
size_t bn_active_memory() { return get_active_memory(); }
size_t bn_peak_memory() { return get_peak_memory(); }
size_t bn_cache_memory() { return get_cache_memory(); }
void bn_reset_peak_memory() { reset_peak_memory(); }
int bn_initialize(const char* metallib_path) { TRY {
    metal::set_metallib_path(metallib_path);
    if (!metal::is_available()) throw std::runtime_error("MLX Metal GPU unavailable");
    set_default_device(Device(Device::gpu, 0)); return 0;
} FAIL_INT }
void bn_free(array* a) { delete a; }
array* bn_f32(const float* data, const int* dims, int rank) { TRY { return new array(data, shape(dims, rank), float32); } FAIL_NULL }
array* bn_u32(const uint32_t* data, const int* dims, int rank) { TRY { return new array(data, shape(dims, rank), uint32); } FAIL_NULL }
int bn_eval(array* a) { TRY { a->eval(); return 0; } FAIL_INT }
int bn_read_bits(array* a, uint32_t* output, size_t count) { TRY {
    a->eval(); if (a->size()!=count) throw std::runtime_error("trace size mismatch");
    if(a->dtype()==uint32) std::memcpy(output,a->data<uint32_t>(),count*4);
    else if(a->dtype()==int32) std::memcpy(output,a->data<int32_t>(),count*4);
    else throw std::runtime_error("trace dtype mismatch");return 0;
} FAIL_INT }
int bn_read(array* a, float* output, size_t count) { TRY {
    if (a->dtype() != float32 || a->size() != count) throw std::runtime_error("invalid logits output shape/dtype");
    a->eval(); std::memcpy(output, a->data<float>(), count * sizeof(float)); return 0;
} FAIL_INT }
array* bn_norm(const array* a, const array* weight, float eps, int head_dim) { TRY {
    auto x = head_dim ? reshape(*a, {-1, head_dim}) : *a;
    return new array(reshape(fast::rms_norm(x, *weight, eps), a->shape()));
} FAIL_NULL }
array* bn_add(const array* a, const array* b) { TRY { return new array(add(*a, *b)); } FAIL_NULL }
array* bn_silu_mul(const array* gate, const array* up) { TRY { return new array(multiply(multiply(*gate, sigmoid(*gate)), *up)); } FAIL_NULL }
int bn_pack(const array* x, int groups, array** outputs) { TRY {
    static auto kernel = fast::metal_kernel("bonsai_pack_signmag_g32", {"x"}, {"planes", "scales", "stats"}, pack_source);
    auto r = kernel({*x}, {{groups, 4}, {groups}, {groups, 2}}, {uint32, float32, int32},
        {groups * 32, 1, 1}, {128, 1, 1}, {{"G", groups}}, {}, false, {});
    auto a = std::make_unique<array>(r[0]), b = std::make_unique<array>(r[1]), c = std::make_unique<array>(r[2]);
    outputs[0] = a.release(); outputs[1] = b.release(); outputs[2] = c.release(); return 0;
} FAIL_INT }
array* bn_project(const array* w, const array* ws, const array* wb, const array* planes, const array* scales, const array* stats, int n, int g) { TRY {
    static auto kernel = fast::metal_kernel("bonsai_w1a4_signmag", {"weights", "weight_scales", "weight_biases", "planes", "scales", "stats"}, {"y"}, project_source);
    return new array(kernel({*w, *ws, *wb, *planes, *scales, *stats}, {{n}}, {float32},
        {n, 1, 1}, {128, 1, 1}, {{"N", n}, {"G", g}}, {}, false, {})[0]);
} FAIL_NULL }
array* bn_embed(const array* w, const array* ws, const array* wb, const array* token, int n, int k) { TRY {
    static auto kernel = fast::metal_kernel("bonsai_w1_embedding", {"weights", "weight_scales", "weight_biases", "token"}, {"y"}, embed_source);
    return new array(kernel({*w, *ws, *wb, *token}, {{k}}, {float32},
        {k, 1, 1}, {128, 1, 1}, {{"N", n}, {"K", k}}, {}, false, {})[0]);
} FAIL_NULL }
array* bn_rope(const array* x, const array* frequencies, const array* scale, const array* position, int nh, int hd) { TRY {
    static auto kernel = fast::metal_kernel("bonsai_yarn_rope", {"x", "frequencies", "attention_scale", "position"}, {"y"}, rope_source);
    return new array(kernel({*x, *frequencies, *scale, *position}, {{nh * hd}}, {float32},
        {nh * hd / 2, 1, 1}, {128, 1, 1}, {{"NH", nh}, {"HD", hd}}, {}, false, {})[0]);
} FAIL_NULL }
array* bn_cache_new(int nk, int capacity, int hd) { TRY { return new array(zeros({1, nk, capacity, hd}, float32)); } FAIL_NULL }
array* bn_cache_grow(const array* old, int nk, int extra, int hd) { TRY {
    return new array(concatenate({*old, zeros({1, nk, extra, hd}, float32)}, 2));
} FAIL_NULL }
array* bn_cache_put(const array* old, const array* value, int nk, int pos, int hd) { TRY {
    return new array(slice_update(*old, reshape(*value, {1, nk, 1, hd}), {0,0,pos,0}, {1,nk,pos+1,hd}));
} FAIL_NULL }
array* bn_attention(const array* q, const array* k, const array* v, int nq, int nk, int seq, int hd) { TRY {
    auto keys = slice(*k, {0,0,0,0}, {1,nk,seq,hd});
    auto values = slice(*v, {0,0,0,0}, {1,nk,seq,hd});
    return new array(reshape(fast::scaled_dot_product_attention(reshape(*q, {1,nq,1,hd}), keys, values, 1.0f/std::sqrt(float(hd))), {nq*hd}));
} FAIL_NULL }
}
