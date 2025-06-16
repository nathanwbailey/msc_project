# Part 4: Improved Decoder

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder/
├── autoencoder_l1_l2/
├── autoencoder_l1_l2_sampling/
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
  Autoencoder baseline with improved deocder

- **autoencoder_l1_l2/**  
  Autoencoder baseline with improved deocder with added L1 and L2 weight decay

- **autoencoder_l1_l2_sampling/**  
  Autoencoder baseline with improved deocder with added L1 and L2 weight decay and hard negative sampling approach

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model using encoder representations directly.

- **latent_classification_model/**  
  Latent Classification Model. 

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative modelling. 

- **neural_ode/**  
  Neural ODE-based latent model to evaluate the temporal smoothness of the latent space 

- **simclr_decoder_improved/**  
  Uses the larger decoder for SIMCLR

- **simclr_decoder_improved_mse_loss/**  
  Uses a larger SIMCLR decoder with MSE losses for each mode. 

- **simclr_decoder_improved_mse_loss_decoded/**   
  Decodes the SIMCLR embeddings in the fine-tuning process as well as the MAE embeddings

- **simclr_decoder_larger_improved_mse_loss_weighted_losses/**  
  Weights losses for wind modes due to higher reconstruction loss.

- **simclr_decoder_weight_decay/**  
  Adds weight decay to downstream LSTM model, keep model same as in part 3.


## Purpose

- Develop and benchmark improved decoder architectures
- Integrate new tasks such as data assimilation and conditional latent diffusion models for generation.