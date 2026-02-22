# Hybrid Machine Learning and Optimization based Spatial Filtering Framework for Robust Sensor Array Signal Processing

## Description
This project implements a hybrid beamforming framework combining:
- Machine Learning (Linear Regression for DOA estimation)
- MVDR beamforming
- Hybrid optimization

The goal is robust sensor array signal processing using a dataset of antenna snapshots.

## Features
- DOA estimation (Noisy, Denoised, ML Prediction)
- SNR improvement (Input vs ML vs MVDR vs Hybrid)
- Beam patterns (ML, MVDR, Hybrid)
- Before/After beamforming signal comparison
- 3D hybrid beam pattern visualization
- SNR vs alpha sweep plot

## Dataset
Put your dataset CSV file inside `dataset/` folder.  
Current example: `C_44_train_converted.csv`

## How to Run
```bash
g++ -std=c++17 hybrid.cpp -I path_to_eigen -o hybrid
./hybrid
