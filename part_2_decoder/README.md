# Part 2: Decoder

## Structure

```
part_2_decoder/
├── save_data.py
├── autoencoder/
├── autoencoder_GN/
├── barlow_twins_decoder/
├── barlow_twins_decoder_and_group_norm/
├── downstream_model_lstm_no_decoder/
├── simclr_decoder/
├── simclr_decoder_group_norm/
```

## Contents

- **save_data.py**  
  Utility script for saving ERA5 data to torch tensor.

- **autoencoder/**  
  Simple Masked Autoencoder for comparison.

- **autoencoder_GN/**  
  Simple Masked Autoencoder with group normalisation for comparison.

- **barlow_twins_decoder/**  
  Adds decoder to barlow twins that reconstructs masked data - uses BN. 

- **barlow_twins_decoder_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for group (instance) norm

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model to test encoder representations directly.

- **simclr_decoder/**  
  Adds decoder to SIMCLR that reconstructs masked data - uses BN.

- **simclr_decoder_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for group (instance) norm

## Purpose

- Provides a direct comparison between autoencoder and contrastive learning approaches.
- Evaluates downstream forecasting performance of different approaches.
- Explore the impact of normalization (batch, group, layer, or none) on downstream task.


## Usage

- Refer to the scripts and notebooks in each subfolder for specific experiments.
