# Flash Attention in CUDA from Scratch

This repository contains a step-by-step implementation of Flash Attention in CUDA. The project progresses from basic GPU primitives to a fully fused, IO-aware, causal attention kernel.

By fusing the attention mechanism and implementing online softmax, this kernel avoids materializing the O(N^2) attention matrix in global memory. Instead, it handles intermediate calculations directly in shared memory. Built as part of the Deep-ML curriculum.

## Concepts Covered
* **Memory Management:** Utilizing shared memory (SRAM) to minimize global memory (VRAM) read/write bottlenecks.
* **Online Softmax:** Computing softmax in chunks to maintain numerical stability without requiring the global sequence maximum upfront.
* **Thread Synchronization:** Implementing grid-stride loops, memory coalescing, and block-level synchronization.
* **Causal Masking:** Applying autoregressive masks for left-to-right sequence generation.

## Usage

Ensure the CUDA Toolkit and `nvcc` are installed, then run the testing scaffold:

```bash
python scaffold.py
```

## Implementation Steps

The kernel development is broken down into 26 distinct steps across five phases.

### Phase 1: CUDA & Linear Algebra Primitives
Basic GPU operations for matrix math using 1D and 2D grid-stride loops.

- [x] 1. `vector_add`
- [x] 2. `scale_array`
- [x] 3. `elementwise_exp`
- [x] 4. `row_max`
- [x] 5. `row_sum`
- [x] 6. `dot_product`
- [x] 7. `matmul`
- [x] 8. `transpose`

### Phase 2: Naive Attention Baseline
Standard O(N^2) attention implemented as a baseline to verify mathematical correctness.

- [x] 9. `qk_scores`
- [x] 10. `softmax_rows`
- [x] 11. `pv_matmul`
- [x] 12. `naive_attention`

### Phase 3: Online Softmax Mathematics
Functions to handle the streaming algebraic corrections required to calculate softmax in blocks.

- [x] 13. `online_max`
- [x] 14. `correction_factor`
- [x] 15. `update_running_sum`
- [x] 16. `rescale_output`

### Phase 4: IO-Aware Tiling & Shared Memory
Memory management functions to load Query, Key, and Value blocks cooperatively and perform local tile math.

- [x] 17. `load_tile`
- [x] 18. `tile_scores`
- [x] 19. `tile_rowmax`
- [x] 20. `tile_exp`
- [x] 21. `tile_rowsum`
- [x] 22. `accumulate_pv`

### Phase 5: Fused Flash Attention
Integrating the shared memory tiles and online softmax into a single kernel launch.

- [x] 23. `flash_attention_kernel`
- [x] 24. `flash_attention_launcher`
- [x] 25. `causal_mask`
- [x] 26. `flash_attention_causal_kernel`
