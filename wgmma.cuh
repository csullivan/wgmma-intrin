#include <cuda_bf16.h>

#include <cassert>
#include <cstdint>

__device__ void ptx_wgmma_bf16_m64n256k16(float r[128], uint64_t desc_a, uint64_t desc_b) {
  int scale_d = false;
  int imm_scale_a = 1;
  int imm_scale_b = 1;
  int imm_trans_a = 0;
  int imm_trans_b = 0;
  asm volatile(
      ".reg .pred %%p;\n\t"
      "setp.eq.s32 %%p, %130, 1;\n\t"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, "
      "%72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, "
      "%88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, "
      "%104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127"
      "}, "
      "%128, "
      "%129, "
      "%%p, 1, 1, 0, 0;"
      : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]),
        "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]),
        "=f"(r[16]), "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]),
        "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31]),
        "=f"(r[32]), "=f"(r[33]), "=f"(r[34]), "=f"(r[35]), "=f"(r[36]), "=f"(r[37]), "=f"(r[38]), "=f"(r[39]),
        "=f"(r[40]), "=f"(r[41]), "=f"(r[42]), "=f"(r[43]), "=f"(r[44]), "=f"(r[45]), "=f"(r[46]), "=f"(r[47]),
        "=f"(r[48]), "=f"(r[49]), "=f"(r[50]), "=f"(r[51]), "=f"(r[52]), "=f"(r[53]), "=f"(r[54]), "=f"(r[55]),
        "=f"(r[56]), "=f"(r[57]), "=f"(r[58]), "=f"(r[59]), "=f"(r[60]), "=f"(r[61]), "=f"(r[62]), "=f"(r[63]),
        "=f"(r[64]), "=f"(r[65]), "=f"(r[66]), "=f"(r[67]), "=f"(r[68]), "=f"(r[69]), "=f"(r[70]), "=f"(r[71]),
        "=f"(r[72]), "=f"(r[73]), "=f"(r[74]), "=f"(r[75]), "=f"(r[76]), "=f"(r[77]), "=f"(r[78]), "=f"(r[79]),
        "=f"(r[80]), "=f"(r[81]), "=f"(r[82]), "=f"(r[83]), "=f"(r[84]), "=f"(r[85]), "=f"(r[86]), "=f"(r[87]),
        "=f"(r[88]), "=f"(r[89]), "=f"(r[90]), "=f"(r[91]), "=f"(r[92]), "=f"(r[93]), "=f"(r[94]), "=f"(r[95]),
        "=f"(r[96]), "=f"(r[97]), "=f"(r[98]), "=f"(r[99]), "=f"(r[100]), "=f"(r[101]), "=f"(r[102]), "=f"(r[103]),
        "=f"(r[104]), "=f"(r[105]), "=f"(r[106]), "=f"(r[107]), "=f"(r[108]), "=f"(r[109]), "=f"(r[110]), "=f"(r[111]),
        "=f"(r[112]), "=f"(r[113]), "=f"(r[114]), "=f"(r[115]), "=f"(r[116]), "=f"(r[117]), "=f"(r[118]), "=f"(r[119]),
        "=f"(r[120]), "=f"(r[121]), "=f"(r[122]), "=f"(r[123]), "=f"(r[124]), "=f"(r[125]), "=f"(r[126]), "=f"(r[127])
      : "l"(desc_a),
        "l"(desc_b),
        "r"(scale_d),
        "r"(imm_scale_a),
        "r"(imm_scale_b),
        "r"(imm_trans_a),
        "r"(imm_trans_b));
}

__device__ void ptx_wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;");
}

__device__ void ptx_wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned 1;");
}

__device__ uint64_t descriptor(
    uint64_t addr,
    uint64_t leading_offset,
    uint64_t stride_offset,
    uint64_t base_offset,
    uint64_t swizzle) {
    /*
        9.7.16.5.1.2.5. Strides
        Leading dimension byte offset of matrix A or B is the distance, in bytes, between two adjacent core matrices in the K dimension.
        Stride dimension byte offset of matrix A or B is the distance, in bytes, between two adjacent core matrices in the M or N dimension.
    */

  return ((addr & 0x3FFFF) >> 4) | ((leading_offset >> 4) << 16) |
      ((stride_offset >> 4) << 32) | (base_offset << 49) | (swizzle << 62);
}

__global__ void wgmma_bf16_m64n256k16_kernel(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* c) {
  constexpr int a_size = 64 * 16;
  constexpr int b_size = 256 * 16;
  __shared__ __nv_bfloat16 a_shared[64 * 16];
  __shared__ __nv_bfloat16 b_shared[256 * 16];

  auto tid = threadIdx.x;
  auto bdim = blockDim.x;

  // This loads exactly 8 registers per thread as the PTX requires 
  // c.f. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16
  // num_regs = shape 64*16 = 1024 elements / 128 threads = 8
  for (int i = tid; i < a_size; i += bdim) {
    a_shared[i] = a[i];
  }

  for (int i = tid; i < b_size; i += bdim) {
    b_shared[i] = b[i];
  }

  float c_regs[128] = {0.0f};
  ptx_wgmma_bf16_m64n256k16(
      c_regs,
      descriptor((uint64_t)a_shared, 128, 256, 0, 0),
      descriptor((uint64_t)b_shared, 128, 128 * 16 / 8, 0, 0));

  ptx_wgmma_commit_group();
  ptx_wgmma_wait_group();

  for (int i = 0; i < 128; i++) {
    c[tid * bdim + i] = __float2bfloat16(c_regs[i]);
  }
}

void wgmma_bf16_m64n256k16(void* a, void* b, void* c, int m, int n, int k) {
  if (m != 64 || n != 256 || k != 16) {
    printf("Error: Current implementation only supports m=64, n=256, k=16\n");
    printf("Received: m=%d, n=%d, k=%d\n", m, n, k);
    assert(false);
  }
  
  // TODO(csullivan): Implement outer loops to support larger matrix sizes

  wgmma_bf16_m64n256k16_kernel<<<1, 128>>>(
      static_cast<__nv_bfloat16*>(a),
      static_cast<__nv_bfloat16*>(b),
      static_cast<__nv_bfloat16*>(c));
}
