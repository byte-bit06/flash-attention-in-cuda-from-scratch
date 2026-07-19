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

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, 
                            float* s_tile, int tile_q, int tile_k, 
                            int head_dim, float scale, 
                            int thread_id, int num_threads) {
    
    // The total number of relationship scores we need to calculate for this tile
    int total_scores = tile_q * tile_k;

    // Teamwork: Threads cooperatively divide the workload
    for (int i = thread_id; i < total_scores; i += num_threads) {
        
        // Find exactly which Query (row) and which Key (column) this thread is comparing
        int q_row = i / tile_k;
        int k_row = i % tile_k;

        float dot_product = 0.0f;
        
        // The Inner Loop: Calculate the dot product across the head dimension
        for (int d = 0; d < head_dim; d++) {
            // Read from the Query tile
            float q_val = q_tile[q_row * head_dim + d];
            
            // Read from the Key tile (Notice we use k_row, NOT a column index. This is the Transpose trick!)
            float k_val = k_tile[k_row * head_dim + d];
            
            dot_product += q_val * k_val;
        }

        // Multiply by the temperature scale and write the final score to shared memory
        s_tile[i] = dot_product * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, 
                            int tile_q, int tile_k, 
                            int thread_id, int num_threads) {
                            
    // The threads cooperatively divide the workload by ROW, not by individual element
    for (int r = thread_id; r < tile_q; r += num_threads) {
        

        float current_max = -INFINITY;
        
        // Scan across every column in this specific row
        for (int c = 0; c < tile_k; c++) {
            float val = s_tile[r * tile_k + c];
            if (val > current_max) {
                current_max = val;
            }
        }
        
        // Write the highest value found into the output array for this row
        row_max_out[r] = current_max;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max, 
                         int tile_q, int tile_k, 
                         int thread_id, int num_threads) {
    
    // The total number of cells in the score tile
    int total_elements = tile_q * tile_k;

    // Teamwork: Threads cooperatively divide the workload cell by cell
    for (int i = thread_id; i < total_elements; i += num_threads) {
        
        // Find exactly which row this specific cell belongs to
        int r = i / tile_k;
        
        // Read the maximum value that we previously calculated for this specific row
        float current_row_max = row_max[r];
        
        // Fetch the raw score, subtract the row's maximum, and exponentiate it in place
        // We use expf() which is the optimized CUDA math function for standard floats
        float raw_score = s_tile[i];
        s_tile[i] = expf(raw_score - current_row_max);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out, 
                            int tile_q, int tile_k, 
                            int thread_id, int num_threads) {
                            
    // The threads cooperatively divide the workload by ROW, just like tile_rowmax
    for (int r = thread_id; r < tile_q; r += num_threads) {
        
        float row_sum = 0.0f;
        
        // Accumulate the sum of all exponentiated values in this row
        for (int c = 0; c < tile_k; c++) {
            row_sum += p_tile[r * tile_k + c];
        }
        
        // Write the final local sum into the output array for this row
        row_sum_out[r] = row_sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, 
                              float* out_acc, int tile_q, int tile_k, 
                              int head_dim, int thread_id, int num_threads) {
    
    // The total number of output cells we are updating
    int total_elements = tile_q * head_dim;

    // Teamwork: Threads cooperatively divide the output matrix cell by cell
    for (int i = thread_id; i < total_elements; i += num_threads) {
        
        // Find exactly which Query row and which Feature column this cell represents
        int q_row = i / head_dim;
        int d_col = i % head_dim;

        float dot_product = 0.0f;
        
        // The Inner Loop: Calculate the dot product between a row of P and a column of V
        for (int k = 0; k < tile_k; k++) {
            // Read from the probability tile (P_tile)
            float p_val = p_tile[q_row * tile_k + k];
            
            // Read from the value tile (V_tile)
            float v_val = v_tile[k * head_dim + d_col];
            
            dot_product += p_val * v_val;
        }

        // CRITICAL: We accumulate (+=) into the existing running output, we do NOT overwrite (=)
        out_acc[i] += dot_product;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    
    // Dynamic Shared Memory (SRAM) Allocation
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = q_tile + (tile_q * head_dim);
    float* v_tile = k_tile + (tile_k * head_dim);
    float* s_tile = v_tile + (tile_k * head_dim);
    float* out_acc = s_tile + (tile_q * tile_k);
    
    // Pointers for local reduction arrays
    float* row_max = out_acc + (tile_q * head_dim);
    float* row_sum = row_max + tile_q;
    
    // Pointers for global Online Softmax tracking state
    float* running_max = row_sum + tile_q;
    float* running_sum = running_max + tile_q;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Global row offset for this specific thread block's Query tile
    int q_row_start = blockIdx.x * tile_q;
    
    // State Initialization
    for (int i = tid; i < tile_q * head_dim; i += num_threads) {
        out_acc[i] = 0.0f;
    }
    for (int r = tid; r < tile_q; r += num_threads) {
        running_max[r] = -1e38f;
        running_sum[r] = 0.0f;
    }
    __syncthreads(); 
    
    // Anchor the Query Tile
    load_tile(q, q_tile, q_row_start, 0, seq_len, head_dim, tile_q, head_dim, tid, num_threads);
    __syncthreads();
    
    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;
    
    // K/V Streaming Loop 
    for (int t = 0; t < num_k_tiles; t++) {
        int k_row_start = t * tile_k;
        
        load_tile(k, k_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        load_tile(v, v_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        __syncthreads();
        
        tile_scores(q_tile, k_tile, s_tile, tile_q, tile_k, head_dim, scale, tid, num_threads);
        __syncthreads();
        
        // Sequence Masking
        for (int i = tid; i < tile_q * tile_k; i += num_threads) {
            int r = i / tile_k;
            int c = i % tile_k;
            if (q_row_start + r >= seq_len || k_row_start + c >= seq_len) {
                s_tile[i] = -1e38f;
            }
        }
        __syncthreads();
        
        tile_rowmax(s_tile, row_max, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        // Online Softmax State Update
        for (int r = tid; r < tile_q; r += num_threads) {
            float m_old = running_max[r];
            float m_block = row_max[r];
            
            // Bypass math for the first tile to prevent 0.0f destruction
            if (m_old < -1e37f) {
                running_max[r] = m_block;
                row_max[r] = m_block;
            } 
            else {
                float m_new = fmaxf(m_old, m_block);
                float alpha = expf(m_old - m_new);
                
                running_max[r] = m_new;
                running_sum[r] *= alpha;
                
                row_max[r] = m_new;
                
                for (int d = 0; d < head_dim; d++) {
                    out_acc[r * head_dim + d] *= alpha;
                }
            }
        }
        __syncthreads();
        
        tile_exp(s_tile, row_max, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        tile_rowsum(s_tile, row_sum, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        for (int r = tid; r < tile_q; r += num_threads) {
            running_sum[r] += row_sum[r];
        }
        __syncthreads();
        
        accumulate_pv(s_tile, v_tile, out_acc, tile_q, tile_k, head_dim, tid, num_threads);
        __syncthreads();
    }
    
    // Epilogue: Final Normalization and Global Memory Write
    for (int i = tid; i < tile_q * head_dim; i += num_threads) {
        int r = i / head_dim;
        int d = i % head_dim;
        int q_idx = q_row_start + r;
        
        if (q_idx < seq_len) {
            out[q_idx * head_dim + d] = out_acc[i] / running_sum[r];
        }
    }
}

# Step 24 - flash_attention_launcher
#include <cmath>

void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    
    // Compute the scaling factor: 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    // Configure Grid and Block dimensions
    // Grid: One block per query tile (ceiling division to handle ragged sequences)
    int grid_size = (seq_len + tile_q - 1) / tile_q;
    
    // Block: 128 threads is the standard workforce we saw in the test logs
    int block_size = 128;
    
    // Calculate dynamic shared memory size in bytes
    // Total Floats = q_tile + k_tile + v_tile + s_tile + out_acc + 4 trackers
    //              = (tq*d) + (tk*d) + (tk*d) + (tq*tk) + (tq*d) + (4*tq)
    //              = 2*(tq*d) + 2*(tk*d) + (tq*tk) + 4*tq
    int num_floats = (2 * tile_q * head_dim) + 
                     (2 * tile_k * head_dim) + 
                     (tile_q * tile_k) + 
                     (4 * tile_q);
                     
    int smem_bytes = num_floats * sizeof(float);
    
    // Launch the kernel
    flash_attention_kernel<<<grid_size, block_size, smem_bytes>>>(
        d_q, d_k, d_v, d_out, 
        seq_len, head_dim, 
        tile_q, tile_k, scale
    );
}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    
    int total_elements = tile_q * tile_k;
    
    // Cooperative grid-stride loop
    for (int i = thread_id; i < total_elements; i += num_threads) {
        // Decode linear index into local 2D tile coordinates
        int local_r = i / tile_k;
        int local_c = i % tile_k;
        
        // Translate local coordinates to absolute global sequence indices
        int global_q = q_row_start + local_r;
        int global_k = k_col_start + local_c;
        
        // If the key is in the "future" relative to the query, blind it
        if (global_k > global_q) {
            s_tile[i] = -INFINITY;
        }
    }
}

# Step 26 - flash_attention_causal_kernel
__global__ void flash_attention_causal_kernel(const float* q, const float* k, const float* v,
                                              float* out, int seq_len, int head_dim,
                                              int tile_q, int tile_k, float scale) {
    
    // Dynamic Shared Memory (SRAM) Allocation
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = q_tile + (tile_q * head_dim);
    float* v_tile = k_tile + (tile_k * head_dim);
    float* s_tile = v_tile + (tile_k * head_dim);
    float* out_acc = s_tile + (tile_q * tile_k);
    
    float* row_max = out_acc + (tile_q * head_dim);
    float* row_sum = row_max + tile_q;
    float* running_max = row_sum + tile_q;
    float* running_sum = running_max + tile_q;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int q_row_start = blockIdx.x * tile_q;
    
    // State Initialization
    for (int i = tid; i < tile_q * head_dim; i += num_threads) {
        out_acc[i] = 0.0f;
    }
    for (int r = tid; r < tile_q; r += num_threads) {
        running_max[r] = -1e38f;
        running_sum[r] = 0.0f;
    }
    __syncthreads(); 
    
    // Anchor the Query Tile
    load_tile(q, q_tile, q_row_start, 0, seq_len, head_dim, tile_q, head_dim, tid, num_threads);
    __syncthreads();
    
    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;
    
    // Causal K/V Streaming Loop 
    for (int t = 0; t < num_k_tiles; t++) {
        int k_row_start = t * tile_k;
        
        // CAUSAL OPTIMIZATION: 
        // If the start of this Key tile is strictly after the end of our Query tile, 
        // then 100% of the scores will be masked out. We can safely skip it and stop iterating!
        if (k_row_start > q_row_start + tile_q - 1) {
            break; 
        }
        
        load_tile(k, k_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        load_tile(v, v_tile, k_row_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        __syncthreads();
        
        tile_scores(q_tile, k_tile, s_tile, tile_q, tile_k, head_dim, scale, tid, num_threads);
        __syncthreads();
        
        // Sequence Masking
        for (int i = tid; i < tile_q * tile_k; i += num_threads) {
            int r = i / tile_k;
            int c = i % tile_k;
            if (q_row_start + r >= seq_len || k_row_start + c >= seq_len) {
                s_tile[i] = -1e38f;
            }
        }
        __syncthreads();
        
        // Causal Masking
        causal_mask(s_tile, q_row_start, k_row_start, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        // Find maximums for online softmax (ignoring -INF)
        tile_rowmax(s_tile, row_max, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        // Online Softmax State Update
        for (int r = tid; r < tile_q; r += num_threads) {
            float m_old = running_max[r];
            float m_block = row_max[r];
            
            // Bypass math for the first tile to prevent NaN destruction
            if (m_old < -1e37f) {
                running_max[r] = m_block;
                row_max[r] = m_block;
            } 
            else {
                float m_new = fmaxf(m_old, m_block);
                float alpha = expf(m_old - m_new);
                
                running_max[r] = m_new;
                running_sum[r] *= alpha;
                
                row_max[r] = m_new;
                
                for (int d = 0; d < head_dim; d++) {
                    out_acc[r * head_dim + d] *= alpha;
                }
            }
        }
        __syncthreads();
        
        tile_exp(s_tile, row_max, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        tile_rowsum(s_tile, row_sum, tile_q, tile_k, tid, num_threads);
        __syncthreads();
        
        for (int r = tid; r < tile_q; r += num_threads) {
            running_sum[r] += row_sum[r];
        }
        __syncthreads();
        
        accumulate_pv(s_tile, v_tile, out_acc, tile_q, tile_k, head_dim, tid, num_threads);
        __syncthreads();
    }
    
    // Epilogue
    for (int i = tid; i < tile_q * head_dim; i += num_threads) {
        int r = i / head_dim;
        int d = i % head_dim;
        int q_idx = q_row_start + r;
        
        if (q_idx < seq_len) {
            // Include a microscopic failsafe in the denominator in case an entire row was perfectly zeroed out
            float denom = fmaxf(running_sum[r], 1e-7f);
            out[q_idx * head_dim + d] = out_acc[i] / denom;
        }
    }
}

