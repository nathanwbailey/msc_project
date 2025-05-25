# Part 2: Decoder

## Structure

```
part_2_decoder/
├── save_data.py
├── autoencoder_first_idea/
├── autoencoder_MAE/
├── barlow_twins_decoder/
├── barlow_twins_decoder_first_idea/
├── barlow_twins_decoder_layer_norm/
├── barlow_twins_decoder_no_batch_norm_and_group_norm/
├── downstream_model_lstm_no_decoder/
├── simclr_decoder/
├── simclr_decoder_first_idea/
├── simclr_decoder_freeze_train/
├── simclr_decoder_freeze_train_group/
├── simclr_decoder_group_norm/
├── simclr_decoder_layer_norm/
```

## Contents

- **save_data.py**  
  Utility script for saving ERA5 data to torch tensor.

- **autoencoder_first_idea/**  
  Initial autoencoder-based approach that reconstructs augmented data to original data. 

- **autoencoder_MAE/**  
  Simple Masked Autoencoder (MAE) for comparison.

- **barlow_twins_decoder/**  
  Adds decoder to barlow twins that reconstructs masked data. 

- **barlow_twins_decoder_first_idea/**  
  Adds decoder to barlow twins that reconstructs augmented masked data. 

- **barlow_twins_decoder_layer_norm/**  
  Switches Batch Norm in barlow_twins_decoder for layer norm

- **barlow_twins_decoder_no_batch_norm_and_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for none and group (instance) norm

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model to test encoder representations directly.

- **simclr_decoder/**  
  Adds decoder to SIMCLR that reconstructs masked data. 

- **simclr_decoder_first_idea/**  
  Adds decoder to SIMCLR that reconstructs augmented masked data. 

- **simclr_decoder_freeze_train/**  
  Freezes encoder of simclr_decoder and trains encoder on top

- **simclr_decoder_freeze_train_group/**  
  Freezes encoder of simclr_decoder_group_norm and trains encoder on top

- **simclr_decoder_group_norm/**  
  Switches Batch Norm in barlow_twins_decoder for group (instance) norm

- **simclr_decoder_layer_norm/**  
  Switches Batch Norm in barlow_twins_decoder for layer norm

## Purpose

- Provides a direct comparison between autoencoder and contrastive learning approaches.
- Evaluates downstream forecasting performance of different approaches.
- Explore the impact of normalization (batch, group, layer, or none) on downstream task.


## Usage

- Refer to the scripts and notebooks in each subfolder for specific experiments.
