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
├── simclr_decoder_mode_drop/
├── simclr_decoder_window/
├── simclr_decoder_window_both/
├── simclr_decoder_window_both_cycle/
├── simclr_decoder_window_hard_neg/
├── simclr_decoder_window_hard_neg_dual_loss/
├── simclr_decoder_window_hard_neg_masked/
├── simclr_decoder_window_hard_neg_ratio/
├── simclr_decoder_window_hard_neg_ratio_both/
├── simclr_decoder_window_hard_neg_ratio_both_cycle/
├── simclr_decoder_window_hard_neg_ratio_both_cycle_mode_drop/
```

## Contents

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream models using encoder representations directly.

- **simclr_decoder_group_norm/**  
  Baseline SIMCLR model with group normalization.

- **simclr_decoder_group_norm_hard_neg/**  
  Adds hard negative sampling to the baseline.

- **simclr_decoder_mask_ratio/**  
  Experiments with varying mask ratios to test robustness.

- **simclr_decoder_mode_drop/**  
  Adds mode dropping for further robustness.

- **simclr_decoder_window/**  
  Uses positive samples at time t+delta to exploit temporal structure.

- **simclr_decoder_window_hard_neg/**  
  Combines windowing with hard negative sampling.

- **simclr_decoder_window_hard_neg_dual_loss/**  
  Applies windowing and masked samples at t in separate SIMCLR losses.

- **simclr_decoder_window_hard_neg_masked/**  
  Applies windowing and masked samples at t in the same SIMCLR loss.

- **simclr_decoder_window_hard_neg_ratio/**  
  Combines windowing, hard negative sampling, and varying mask ratios.

- **simclr_decoder_window_hard_neg_ratio_both/**  
  Adds positive pairs at t+delta and t-delta in separate SIMCLR losses.

- **simclr_decoder_window_hard_neg_ratio_both_cycle/**  
  Adds cycle consistency loss to SIMCLR projections.

- **simclr_decoder_window_hard_neg_ratio_both_cycle_mode_drop/**  
  Adds mode dropping to the above approach.

## Purpose

- Test and improve the robustness of decoder models to noisy, masked, or missing data.
- Evaluate the effect of hard negative sampling and windowing on contrastive learning.
- Benchmark against downstream weather forecasting tasks in both latent and original space.