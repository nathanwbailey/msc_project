# Part 4: Improved Decoder

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder_batch_size/
├── autoencoder_hard_neg_batch_size/
├── downstream_model_lstm_no_decoder/
├── latent_classification_model
├── latent_diffusion_model_conditional_attn/
├── neural_ode/
├── simclr_batch_size/
├── simclr_cap_alpha/
├── simclr_cap_alpha_batch_size/
├── simclr_cap_alpha_batch_size_cycle/
├── simclr_cycle_loss/
```

## Contents

- **autoencoder_batch_size/**  
  Autoencoder baseline with 256 batch size

- **autoencoder_hard_neg_batch_size/**  
  Autoencoder baseline with 256 batch size and hard negative sampling approach

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model using encoder representations directly.

- **latent_classification_model/**  
  Latent Classification Model. 

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative modelling. 

- **neural_ode/**  
  Neural ODE-based latent model to evaluate the temporal smoothness of the latent space 

- **simclr_batch_size/**  
  SIMCLR model with 256 batch size

- **simclr_cap_alpha/**  
  SIMCLR model with restricted alpha and 128 batch size

- **simclr_cap_alpha_batch_size/**   
   SIMCLR model with restricted alpha and 256 batch size

- **simclr_cap_alpha_batch_size_cycle/**  
  SIMCLR model with restricted alpha, 256 batch size and tweaked cycle loss

- **simclr_cycle_loss/**  
  SIMCLR model with 128 batch size and tweaked cycle loss


## Purpose

- Analyse the latent space and confirm why certain models offer increased forecasting performance over others.
- Implement changes based ont this analysis,