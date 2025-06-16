# Part 3: Robust

This folder contains experiments focused on improving the robustness and generalization of the SIMCLR group (instance) norm approach.

## Structure

```
part_3_robust/
├── README.md
├── downstream_model_lstm_no_decoder/
├── simclr_decoder_group_norm/
├── simclr_decoder_group_norm_hard_neg/
├── simclr_decoder_mask_ratio/
├── simclr_decoder_window/
├── simclr_decoder_window_hard_neg/
├── simclr_decoder_window_hard_neg_ratio/
├── simclr_decoder_window_hard_neg_ratio_expanded/
├── simclr_decoder_window_hard_neg_ratio_expanded_cycle/
```

## Contents

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream models using encoder representations directly.

- **simclr_decoder_group_norm/**  
  Baseline SIMCLR model with group normalization.

- **simclr_decoder_group_norm_hard_neg/**  
  Adds hard negative sampling to the baseline.

- **simclr_decoder_mask_ratio/**  
  Adds 50%-70% mask ratios to test robustness.

- **simclr_decoder_window/**  
  Uses positive samples at time t+delta to exploit temporal structure.

- **simclr_decoder_window_hard_neg/**  
  Combines windowing with hard negative sampling.

- **simclr_decoder_window_hard_neg_ratio/**  
  Combines windowing, hard negative sampling, and varying mask ratios.

- **simclr_decoder_window_hard_neg_ratio_both/**  
  Adds positive pairs at t+delta and t-delta in separate SIMCLR losses.

- **simclr_decoder_window_hard_neg_ratio_both_cycle/**  
  Adds cycle consistency loss to SIMCLR projections.


## Purpose

- Test and improve the robustness of decoder models to noisy, masked, or missing data.
- Evaluate the effect of hard negative sampling and windowing on contrastive learning.
- Benchmark against downstream weather forecasting tasks in both latent and original space.