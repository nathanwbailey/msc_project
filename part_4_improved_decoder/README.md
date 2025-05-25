# Part 4: Improved Decoder

This folder contains the final and most advanced decoder experiments for the MSc project on multi-modal data fusion through contrastive learning in geoscience. The focus here is on improving decoder architectures, integrating new tasks, and pushing reconstruction and downstream performance.

## Structure

```
part_4_improved_decoder/
├── README.md
├── autoencoder/
│   ├── augment_functions.py
│   ├── dataset.py
│   └── ...
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
└── ...
```

## Contents & Key Experiments

- **autoencoder/**  
  Improved autoencoder baselines and utilities (augmentation, datasets, etc.).

- **autoencoder_l1_group_norm/**  
  Autoencoder experiments with L1 loss and group normalization.

- **simclr_decoder_larger\***  
  Larger and more expressive decoders trained on SimCLR representations, with various architectural tweaks (e.g., added variance, specific decoder heads, cycle weighting, momentum training).

- **simclr_decoder_gan\***  
  Decoder models using adversarial (GAN) training for sharper reconstructions.

- **latent_diffusion_model_conditional_attn/**  
  Conditional latent diffusion models for generative reconstruction and data assimilation.

- **neural_ode/**  
  Neural ODE-based decoder experiments for continuous-time modeling.

- **downstream_model_lstm_no_decoder/**  
  LSTM-based downstream models using encoder representations directly, for comparison.

- **downstream_task_transformer_no_decoder/**  
  Transformer-based downstream models without a decoder.

- **Notebooks (e.g., `eval_autoregressive.ipynb`, `recon_vis_downstream.ipynb`, `visual_window_next_t.ipynb`)**  
  Used for evaluation, visualization, and analysis of reconstructions, embeddings, and downstream forecasting.

## Purpose

- Develop and benchmark improved decoder architectures for multi-modal geoscientific data.
- Compare advanced decoders (e.g., GAN, diffusion, ODE) against autoencoder and SimCLR baselines.
- Integrate new tasks such as data assimilation and conditional generation.
- Evaluate on challenging downstream tasks with extended context and stride.

## Usage

- Each subfolder contains scripts and notebooks for a specific experiment or model variant.
- Main training and evaluation scripts are typically named `main.py` or similar.
- Use the provided notebooks for visualization and in-depth analysis of results.

## Notes

- For experiment-specific details, refer to the README or code comments within each subfolder.
- This part represents the culmination of the project, with the most robust and high-performing models.

---

For questions or collaboration, please contact the repository owner.