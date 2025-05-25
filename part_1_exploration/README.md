# Part 1: Exploration

## Structure

```
part_1_exploration/
├── save_data.py
├── autoencoder/
├── barlow_twins/
├── SCL/
├── supervised_contrastive_learning/
```

## Contents

- **save_data.py**  
  Utility script for saving ERA5 Dataset to torch tensor file.

- **autoencoder/**  


- **barlow_twins/**  
  Experiments and scripts for the Barlow Twins contrastive learning method.

- **SIMCLR/**  
  Experiments and scripts for SIMCLR contrastive learning.

- **supervised_contrastive_learning/**  
  Experiments and scripts for Supervised Contrastive Learning. 

## Purpose

- Compare contrastive learning approaches with autoencoder baseline. Explores and validates different contrastive learning methods (SimCLR, Barlow Twins, Supervised Contrastive Learning).

## Usage

1. Review the scripts in each subfolder for specific experiment details.
2. Run the main scripts (e.g., `autoencoder_main.py`) as needed for training or evaluation.