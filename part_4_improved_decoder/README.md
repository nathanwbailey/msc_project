# Part 4: Improved Decoder

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder/
├── autoencoder_l1_l2/
├── autoencoder_l1_l2_sampling/
├── autoencoder_sampling/
├── downstream_model_lstm_no_decoder/
├── latent_classification_model
├── latent_diffusion_model_conditional_attn/
├── neural_ode/
├── simclr_decoder_improved/
├── simclr_decoder_improved_mse_loss/
├── simclr_decoder_improved_mse_loss_decoded/
├── simclr_decoder_improved_mse_loss_weighted_losses/
├── simclr_decoder_weight_decay/
```

## Contents

- **autoencoder/**  
  Autoencoder baseline with improved decoder

- **autoencoder_l1_l2/**  
  Autoencoder baseline with improved decoder with added L1 and L2 weight decay

- **autoencoder_l1_l2_sampling/**  
  Autoencoder baseline with improved decoder with added L1 and L2 weight decay and hard negative sampling approach

- **autoencoder_sampling/**  
  Autoencoder baseline with an improved decoder with a hard negative sampling approach

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model for forecasting.

- **latent_classification_model/**  
  Latent Classification Model. 

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion model. 

- **neural_ode/**  
  Neural ODE-based latent model to evaluate the temporal smoothness of the latent space 

- **simclr_decoder_improved/**  
  Uses the larger decoder for SIMCLR

- **simclr_decoder_improved_mse_loss/**  
  Uses a larger SIMCLR decoder with MSE losses for each mode. 

- **simclr_decoder_improved_mse_loss_decoded/**   
  Decodes the SIMCLR embeddings in the fine-tuning process.

- **simclr_decoder_larger_improved_mse_loss_weighted_losses/**  
  SIMCLR model with weighted losses for the wind modes.

- **simclr_decoder_weight_decay/**  
  Adds weight decay to the downstream LSTM model, keeping the model the same as in part 3.


## Purpose

- Develop and benchmark improved decoder architectures
- Integrate new tasks such as latent classification, autoregressive forecasting and conditional latent diffusion models for generation.
