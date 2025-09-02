# MSc Project: Multi-Modal Data Fusion Through Contrastive Learning in Geoscience

This repository contains code and experiments for an MSc thesis focused on contrastive learning for multi-modal ERA5 data.

## Repository Structure

```
.
├── part_1_exploration/
├── part_2_decoder/
├── part_3_robust/
├── part_4_improved_decoder/
├── part_5_further_analysis/
├── part_6_multimodal/
├── requirements.txt
```

## Project Parts

- **Part 1: Exploration**
    - Initial experiments and exploration of contrastive learning
    - Trains SIMCLR, Barlow Twins, Supervised Contrastive Learning Methods
    - Contrasts with an Autoencoder Approach

- **Part 2: Decoder**
    - Adds a decoder to the contrastive learning methods to provide a direct comparison to the autoencoder approach
    - Compares Group Norm and Batch Norm Approaches
    - Evaluates the approaches to the task of downstream weather forecasting

- **Part 3: Robust**
    - Attempts to make the chosen SIMCLR approach more robust and have better downstream performance
    - Trialled numerous approaches and collated them all into one final solution

- **Part 4: Improved Decoder**
    - Final decoder architecture improvements
    - Adds additional tasks of autoregressive forecasting, latent classification, and conditional latent diffusion models

- **Part 5: Further Analysis**
    - Analysis on the latent space to link smoothness and forecasting performance
    - Implements changes to batch size, cycle loss and alpha decay based on findings

- **Part 6: Multimodal**
    - Implements several multimodal late fusion methods
    - Average Pooling
    - Self-Attention
    - GNN

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

4. **Download ERA5 data using e.g. `python3 save_data.py` in the part 1 and part 2 directories.**

5. **Replace any paths in main.py to your local ERA5 Pytorch data file**
    e.g. replace `data = torch.load("/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt")`

## Details on Folder Structure

Each model in the directories has corresponding files for training and testing that are self-explanatory.

Each model has several notebooks that provide results in the report:

- eval.ipynb - **Evaluates the model for single-step forecasting**
- eval_autoregressive.ipynb - **Evaluates the model for autoregressive forecasting**
- eval_autoregressive_seed_avg.ipynb -  **Evaluates the model for autoregressive forecasting for strided data**
- eval_latent.ipynb - **Evaluates the model for conditional latent diffusion**
- visual.ipynb - **Visualises the latent space and computes smoothness metrics**
- visual_window_next_t.ipynb - **Plots trajectories of context windows with the next step**


## Notes

- Each subfolder contains scripts and modules for the specific experiments and model variants.

---
If you have any questions or collaboration, please feel free to reach out to me.
