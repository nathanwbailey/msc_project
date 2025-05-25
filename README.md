# MSc Project: Multi-Modal Data Fusion Through Contrastive Learning in Geoscience

This repository contains code and experiments for an MSc thesis focused on contrastive learning and decoder design for multi-modal geoscientific data.

## Repository Structure

```
.
├── part_1_exploration/
│   ├── save_data.py
│   ├── autoencoder/
│   ├── barlow_twins/
│   ├── SCL/
│   └── supervised_contrastive_learning/
├── part_2_decoder/
│   ├── save_data.py
│   ├── autoencoder_first_idea/
│   ├── autoencoder_MAE/
│   ├── barlow_twins_decoder/
│   ├── barlow_twins_decoder_first_idea/
│   ├── barlow_twins_decoder_layer_norm/
│   ├── barlow_twins_decoder_no_batch_norm_and_group_norm/
│   ├── downstream_model_lstm_no_decoder/
│   ├── simclr_decoder/
│   └── simclr_decoder_first_idea/
├── part_3_robust/
├── part_4_improved_decoder/
├── requirements.txt
```

## Project Parts

- **Part 1: Exploration**
    - Initial experiments and exploration of contrastive learning
    - Trains SIMCLR, Barlow Twins, Supervised Contrastive Learning Methods
    - Contrasts with an Autoencoder Approach

- **Part 2: Decoder**
    - Adds a decoder to the contrastive learning methods to provide a direct comparison to the autoencoder approach
    - Contrasts 2 methods of adding the decoder
    - Compares Instance Norm and Batch Norm Approaches
    - Evaluates the approaches on the task of downstream weather forecasting

- **Part 3: Robust**
    - Attempts to make the chosen SIMCLR approach more robust and have better downstream performance
    - Trialled numerous approaches and collated them all into one final solution

- **Part 4: Improved Decoder**
    - Final decoder architecture improvements
    - Contrasts final approach with a more representative autoencoder
    - Adds additional tasks of data assimilation and conditional latent diffusion models
    - Expands the downstream task of forecasting with added stride and changing context windows 

## Getting Started

1. **Clone the repository**
    ```
    git clone <repo-url>
    ```

2. **Create and activate a virtual environment**
    ```
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

## Notes

- Each subfolder contains scripts and modules for specific experiments or model variants.

---
For questions or collaboration, please get in touch with me
