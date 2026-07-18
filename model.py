"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Calculate this specific thread's global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<n){
        a[i] = a[i] * scalar;
    };


}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        a[i] = exp(a[i]);
    };
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int R, int C) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    // Protect against the tail problem
    if (r < R) {        // To find a 2D coordinate (r, c) in a flat 1D array, use: (r * total_columns) + c
        float max_val = matrix[r * C + 0];

        // Loop through the rest of the columns in this specific row
        for (int c = 1; c < C; c++) {
            float current_val = matrix[r * C + c];
            if (current_val > max_val) {
                max_val = current_val;
            }
        }

        // Write the final max value to the output array
        out[r] = max_val;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int R, int C) {
    extern __shared__ float sdata[];

    int r = blockIdx.x; 
    int tid = threadIdx.x;

    if (r < R) {
        float sum = 0.0f;
        for (int c = tid; c < C; c += blockDim.x) {
            sum += matrix[r * C + c];
        }
        
        sdata[tid] = sum;
        
        __syncthreads(); 

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out[r] = sdata[0];
        }
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

# Step 7 - matmul (not yet solved)
# TODO: implement

# Step 8 - transpose (not yet solved)
# TODO: implement

# Step 9 - qk_scores (not yet solved)
# TODO: implement

# Step 10 - softmax_rows (not yet solved)
# TODO: implement

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

