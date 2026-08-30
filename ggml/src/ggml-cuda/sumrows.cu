#include "reduce_rows.cuh"
#include "sumrows.cuh"

static __global__ void sum_rows_4_f32(const float * x, float * dst, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + int64_t(row) * 4;
    dst[row] = (xr[0] + xr[2]) + (xr[1] + xr[3]);
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    if (ncols == 4 && nrows <= INT_MAX) {
        constexpr int threads = 256;
        const int blocks = (nrows + threads - 1) / threads;
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, stream);
        ggml_cuda_kernel_launch(sum_rows_4_f32, launch_params, x, dst, nrows);
        return;
    }
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (ncols == 4 && nrows <= INT_MAX) {
        constexpr int threads = 256;
        const int blocks = ((int) nrows + threads - 1) / threads;
        const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, stream);
        ggml_cuda_kernel_launch(sum_rows_4_f32, launch_params, src0_d, dst_d, (int) nrows);
        return;
    }

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    }
}
