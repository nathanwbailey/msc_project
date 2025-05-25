# Part 4: Improved Decoder

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder/
├── autoencoder_l1_group_norm/
├── downstream_model_lstm_no_decoder/
├── downstream_task_transformer_no_decoder/
├── latent_diffusion_model_conditional_attn/
├── neural_ode/
├── simclr_decoder_gan/
├── simclr_decoder_gan_end/
├── simclr_decoder_larger/
├── simclr_decoder_larger_specific_decoder/
├── simclr_decoder_larger_specific_decoder_added_variance/
├── simclr_decoder_larger_specific_decoder_first_idea/
├── simclr_decoder_larger_specific_decoder_first_idea_added_variance/
├── simclr_decoder_larger_specific_decoder_first_idea_cycle_emb/
├── simclr_decoder_larger_specific_decoder_first_idea_diff_training/
├── simclr_decoder_larger_specific_decoder_momentum_cycle/
├── simclr_decoder_larger_specific_decoder_first_idea_momentum_train/
├── simclr_decoder_specific_losses/
├── simclr_decoder_larger_specific_decoder_res/
├── simclr_decoder_larger_specific_decoder_se/
├── simclr_decoder_larger_specific_decoder_train_flip_l1_decay/
├── simclr_decoder_larger_specific_decoder_weighted_losses/
├── simclr_decoder_perceptual_loss/
├── simclr_decoder_shuffle_lstm/
├── simclr_decoder_specific_losses/
├── simclr_decoder_ssim/
├── simclr_decoder_transformer/
├── simclr_decoder_weight_decay/
```

## Contents

- **autoencoder/**  
  More representative autoencoder baseline

- **autoencoder_l1_group_norm/**  
  More representative autoencoder baseline with added L1 and L2 weight decay and instance norm instead of batch norm

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model using encoder representations directly.

- **downstream_task_transformer_no_decoder/**  
  Transformer-based downstream model using encoder representations directly.

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative modelling. 

- **neural_ode/**  
  Neural ODE-based latent model to evaluate the temporal smoothness of the latent space 

- **simclr_decoder_gan/**  
  Adds a GAN to the decoder.

- **simclr_decoder_gan_end/**  
  Adds a GAN to the decoder, in the frozen training stage only.

- **simclr_decoder_larger/**  
  Uses a larger SIMCLR decoder

- **simclr_decoder_larger_specific_decoder/**  
  Uses a larger SIMCLR decoder with MSE losses for each mode. 

- **simclr_decoder_larger_specific_decoder_added_variance/**  
  Adds additional variance and co-variance losses to SIMCLR

- **simclr_decoder_larger_specific_decoder_first_idea/**  
  Decodes the SIMCLR embeddings in the fine-tuning process as well as the MAE embeddings

- **simclr_decoder_larger_specific_decoder_first_idea_added_variance/**  
  Adds additional variance and co-variance losses

- **simclr_decoder_larger_specific_decoder_first_idea_cycle_emb/**  
  Adds Cycle Consistency loss to the embeddings rather than the projections

- **simclr_decoder_larger_specific_decoder_first_idea_diff_training/**  
  Changes training approach to nested approach.

- **simclr_decoder_larger_specific_decoder_first_idea_momentum_train/**  
  Changes training approach to momentum enocoder approach.

- **simclr_decoder_larger_specific_decoder_weighted_losses/**  
  Weights losses for wind modes due to higher reconstruction loss.

- **simclr_decoder_larger_specific_decoder_res/**  
  Adds ResNet branch to wind losses to attempt to reduce reconstruction loss for those modes

- **simclr_decoder_larger_specific_decoder_se/**  
  Adds Squeeze-Excite branch to wind losses to attempt to reduce reconstruction loss for those modes

- **simclr_decoder_larger_specific_decoder_train_flip_l1_decay/**  
  Flips the training process and also evaluates the autoencoder with the hard negative sampling approach

- **simclr_decoder_perceptual_loss/**  
  Adds perceptual loss to the training of the MAE

- **simclr_decoder_shuffle_lstm/**  
  Shuffles the data before inputting into the downstream model to show the temporal structure created by the SIMCLR model

- **simclr_decoder_specific_losses/**  
  Adds additional losses for wind modes to attempt to reduce reconstruction loss for those modes

- **simclr_decoder_ssim/**  
  Adds SSIM loss to the training of the MAE

- **simclr_decoder_transformer/**  
  Trains a downstream transformer based model instead of LSTM

- **simclr_decoder_weight_decay/**  
  Adds weight decay to downstream LSTM model.





## Purpose

- Develop and benchmark improved decoder architectures
- Integrate new tasks such as data assimilation and conditional latent diffusion models for generation.