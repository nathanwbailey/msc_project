# Part 2: Decoder

## Structure

```
part_2_decoder/
├── save_data.py
├── autoencoder_MAE/
├── autoencoder_MAE_GN/
├── barlow_twins_decoder/
├── barlow_twins_decoder_no_batch_norm_and_group_norm/
├── downstream_model_lstm_no_decoder/
├── simclr_decoder/
├── simclr_decoder_final_stage/
├── simclr_decoder_group_norm/
├── simclr_decoder_group_norm_final_stage/
```

## Contents

- **save_data.py**  
  Utility script for saving ERA5 data to torch tensor.

- **autoencoder_MAE/**  
  Simple Masked Autoencoder (MAE) for comparison.

- **autoencoder_MAE_GN/**  
  Simple Masked Autoencoder (MAE) with group normalisation for comparison.

- **barlow_twins_decoder/**  
  Adds decoder to barlow twins that reconstructs masked data - uses BN. 

- **barlow_twins_decoder_no_batch_norm_and_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for none and group (instance) norm

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model to test encoder representations directly.

- **simclr_decoder/**  
  Adds decoder to SIMCLR that reconstructs masked data - uses BM. 

- **simclr_decoder_final_stage/**  
  Freezes encoder of simclr_decoder and trains encoder on top - for the final training stage

- **simclr_decoder_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for group (instance) norm

- **simclr_decoder_group_norm_final_stage/**  
  Freezes encoder of simclr_decoder_group_norm and trains encoder on top - for the final training stage

## Purpose

- Provides a direct comparison between autoencoder and contrastive learning approaches.
- Evaluates downstream forecasting performance of different approaches.
- Explore the impact of normalization (batch, group, layer, or none) on downstream task.


## Usage

- Refer to the scripts and notebooks in each subfolder for specific experiments.
