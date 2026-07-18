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

# Step 7 - matmul
__global__ void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // X-axis 
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Y-axis

    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }

        C[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {

        int in_idx = r * cols + c;

        int out_idx = c * rows + r;

        out[out_idx] = in[in_idx];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* Q, const float* K, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < seq_len && j < seq_len) {
        const float* query_row = Q + (i * head_dim);
        const float* key_row   = K + (j * head_dim);

        float dot_prod = dot_product(query_row, key_row, head_dim);

        scores[i * seq_len + j] = dot_prod / sqrtf((float)head_dim); 
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    // Assign one block to one specific row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Protect against the tail problem
    if (row >= rows) return;

    // Find the starting memory address for this specific row
    float* row_ptr = matrix + row * cols;

    __shared__ float sdata[1024];

    // FIND THE MAXIMUM VALUE (row_max)
    float local_max = -1e20f; // Start with an extremely small number
    
    // Each thread finds the max of its assigned columns
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Tournament bracket fold to find the absolute maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0]; // Thread 0 now holds the max


    // SUBTRACT MAX, EXPONENTIATE, & SUM
    float local_sum = 0.0f;
    
    // Each thread subtracts max, applies expf(), and sums its assigned columns
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(row_ptr[i] - row_max);
        row_ptr[i] = val; // Overwrite matrix with exp values
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Tournament bracket fold to find the total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0]; // Thread 0 now holds the sum


    // NORMALIZE (DIVIDE BY SUM)
    for (int i = tid; i < cols; i += blockDim.x) {
        row_ptr[i] /= row_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* P, const float* V, float* out, int seq_len, int head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < head_dim) {
        float sum = 0.0f;
        
        for (int j = 0; j < seq_len; j++) {
            float p_val = P[row * seq_len + j];
            float v_val = V[j * head_dim + col];
            
            sum += p_val * v_val;
        }
        
        out[row * head_dim + col] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* Q, const float* K, const float* V, float* d_out, int seq_len, int head_dim) {
    // Allocate intermediate scratch memory for the P matrix (seq_len x seq_len)
    float* d_scores;
    size_t scores_size = seq_len * seq_len * sizeof(float);
    cudaMalloc((void**)&d_scores, scores_size);

    // qk_scores (Q * K^T)
    // 2D grid covering every (row, col) pair in the (seq_len, seq_len) score matrix
    dim3 block_qk(16, 16);
    dim3 grid_qk((seq_len + block_qk.x - 1) / block_qk.x, 
                 (seq_len + block_qk.y - 1) / block_qk.y);
    
    qk_scores<<<grid_qk, block_qk>>>(Q, K, d_scores, seq_len, head_dim);

    // softmax_rows (Probabilities)
    // 1D grid with 1 block per row. 
    // The block size MUST be a power of 2 for the tree reduction to work safely.
    int threads_softmax = 256; 
    
    // The instructions warn about dynamic shared memory. We pass it as the 3rd launch parameter.
    size_t shared_mem_bytes = seq_len * sizeof(float); 
    
    softmax_rows<<<seq_len, threads_softmax, shared_mem_bytes>>>(d_scores, seq_len, seq_len);

    // pv_matmul (P * V)
    // 2D grid covering every (row, col) pair in the output matrix (seq_len, head_dim)
    dim3 block_pv(16, 16);
    dim3 grid_pv((head_dim + block_pv.x - 1) / block_pv.x, 
                 (seq_len + block_pv.y - 1) / block_pv.y);
    
    pv_matmul<<<grid_pv, block_pv>>>(d_scores, V, d_out, seq_len, head_dim);

    // Free the scratch memory to prevent memory leaks
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return (old_sum * correction) + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int d = 0; d < head_dim; d++) {
            out_row[d] *= correction;
        }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, 
                          float* shared_dst, 
                          int src_row_start, 
                          int src_col_start, 
                          int src_rows, 
                          int src_cols, 
                          int tile_rows, 
                          int tile_cols, 
                          int thread_id, 
                          int num_threads) {
    
    // Calculate the exact size of the small tile we are moving to fast memory
    int total_elements = tile_rows * tile_cols;

    // Teamwork: Threads stride across the tile to load it cooperatively
    for (int i = thread_id; i < total_elements; i += num_threads) {
        
        // Find exactly where this element belongs in the small tile
        int local_row = i / tile_cols;
        int local_col = i % tile_cols;

        // Map that coordinate to the giant global matrix
        int global_row = src_row_start + local_row;
        int global_col = src_col_start + local_col;

        // Safety check: Does this coordinate actually exist in the big matrix?
        if (global_row >= 0 && global_row < src_rows && 
            global_col >= 0 && global_col < src_cols) {
            // It exists! Copy the real data.
            shared_dst[i] = src[global_row * src_cols + global_col];
        } else {
            // It hangs off the edge. Pad it with harmless zeroes.
            shared_dst[i] = 0.0f;
        }
    }
}

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

