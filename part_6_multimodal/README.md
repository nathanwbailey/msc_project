# Part 4: Improved Decoder

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder_comparison_gnn
├── autoencoder_comparison_self_attention
├── downstream_model_lstm_no_decoder/
├── latent_classification_model
├── latent_diffusion_model_conditional_attn/
├── neural_ode/
├── simclr_multi_branch
├── simclr_multi_branch_gnn
├── simclr_multi_branch_self_attention/
```

## Contents

- **autoencoder_comparison_gnn/**  
  Autoencoder comparison to GNN fusion

- **autoencoder_comparison_self_attention/**  
  Autoencoder comparison to Self-Attention fusion

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream model using encoder representations directly.

- **latent_classification_model/**  
  Latent Classification Model. 

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative modelling. 

- **neural_ode/**  
  Neural ODE-based latent model to evaluate the temporal smoothness of the latent space 

- **simclr_multi_branch/**  
  Multimodal fusion method that uses average pooling

- **simclr_multi_branch_gnn/**  
  Multimodal fusion method that uses GNN

- **simclr_multi_branch_self_attention/**   
   Multimodal fusion method that uses Self-Attention


## Purpose

- Implement and investigate several late-fusion multimodal methods.