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
├── simclr_decoder_larger_specific_decoder_first_idea_cycle_weight/
├── simclr_decoder_larger_specific_decoder_momentum/
├── simclr_decoder_larger_specific_decoder_momentum_cycle/
├── simclr_decoder_larger_specific_decoder_momentum_cycle_added_variance/
├── simclr_decoder_larger_specific_decoder_momentum_cycle_added_variance_group_norm/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_added_variance/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_added_variance/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_added_variance_cycle_weight/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_cycle_weight/
├── simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_weight/
├── simclr_decoder_larger_specific_decoder_momentum_weight/
├── simclr_decoder_larger_specific_decoder_weight/
```

## Contents

- **autoencoder/**  
  Improved autoencoder baselines and utilities (augmentation, datasets, etc.).

- **autoencoder_l1_group_norm/**  
  Autoencoder experiments with L1 loss and group normalization.

- **simclr_decoder_larger/**  
  Larger SimCLR decoders.

- **simclr_decoder_larger_specific_decoder/**  
  SimCLR decoders with specific decoder architectures.

- **simclr_decoder_larger_specific_decoder_added_variance/**  
  SimCLR decoders with added variance in the architecture.

- **simclr_decoder_larger_specific_decoder_first_idea/**  
  Initial ideas for specific SimCLR decoder architectures.

- **simclr_decoder_larger_specific_decoder_first_idea_added_variance/**  
  Initial specific decoder ideas with added variance.

- **simclr_decoder_larger_specific_decoder_first_idea_cycle_weight/**  
  Initial specific decoder ideas with cycle weighting.

- **simclr_decoder_larger_specific_decoder_momentum/**  
  SimCLR decoders with momentum.

- **simclr_decoder_larger_specific_decoder_momentum_cycle/**  
  SimCLR decoders with momentum and cycle.

- **simclr_decoder_larger_specific_decoder_momentum_cycle_added_variance/**  
  SimCLR decoders with momentum, cycle, and added variance.

- **simclr_decoder_larger_specific_decoder_momentum_cycle_added_variance_group_norm/**  
  SimCLR decoders with momentum, cycle, added variance, and group normalization.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm/**  
  SimCLR decoders with momentum and group normalization.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_added_variance/**  
  SimCLR decoders with momentum, group normalization, and added variance.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle/**  
  SimCLR decoders with momentum, group normalization, and cycle.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_added_variance/**  
  SimCLR decoders with momentum, group normalization, cycle, and added variance.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_added_variance_cycle_weight/**  
  SimCLR decoders with momentum, group normalization, cycle, added variance, and cycle weighting.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_cycle_weight/**  
  SimCLR decoders with momentum, group normalization, cycle, and cycle weighting.

- **simclr_decoder_larger_specific_decoder_momentum_group_norm_cycle_weight/**  
  SimCLR decoders with momentum, group normalization, and cycle weighting.

- **simclr_decoder_larger_specific_decoder_momentum_weight/**  
  SimCLR decoders with momentum and weighting.

- **simclr_decoder_larger_specific_decoder_weight/**  
  SimCLR decoders with weighting.

- **simclr_decoder_gan/**  
  Decoder models using adversarial (GAN) training for sharper reconstructions.

- **simclr_decoder_gan_end/**  
  GAN-based decoder models with different training strategies.

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative reconstruction and data assimilation.

- **neural_ode/**  
  Neural ODE-based decoder experiments for continuous-time modeling.

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream models using encoder representations directly, for comparison.

- **downstream_task_transformer_no_decoder/**  
  Transformer-based downstream models without a decoder.

## Purpose

- Develop and benchmark improved decoder architectures
- Integrate new tasks such as data assimilation and conditional latent diffusion models for generation.