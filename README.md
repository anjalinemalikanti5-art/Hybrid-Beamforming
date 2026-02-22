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
##How to run
bash
g++ -std=c++17 hybrid.cpp -I path_to_eigen -o hybrid
./hybrid


# Hybrid Beamforming

**Enhancing Signal-to-Noise Ratio (SNR) in antenna arrays using ML, MVDR, and Hybrid Beamforming.**

---

## Project Overview

Hybrid Beamforming is a signal processing technique used to improve the performance of antenna arrays by combining multiple beamforming approaches. This project focuses on **Direction of Arrival (DOA) estimation** and SNR enhancement using:

- **ML-based DOA prediction**  
- **MVDR (Minimum Variance Distortionless Response) beamforming**  
- **Hybrid optimization of beamforming weights**  

Key objectives:

1. Load and process antenna snapshot data.  
2. Denoise noisy DOA measurements.  
3. Predict DOA using a linear ML model.  
4. Compute covariance matrices and derive beamforming weights.  
5. Optimize hybrid beamforming to maximize SNR.  
6. Visualize results through plots for DOA estimation, beam patterns, and SNR comparison.  

---

## Project Workflow

### Numbered Steps

1. **Data Input:** Load `C_44_train_converted.csv` from the `dataset/` folder.  
2. **DOA Denoising:** Apply Savitzky–Golay filter to smooth noisy angles.  
3. **ML Prediction:** Perform linear regression to estimate DOA per snapshot.  
4. **Covariance Calculation:** Compute covariance matrix of received signals.  
5. **Beamforming:**  
   - Compute **ML beamformer**  
   - Compute **MVDR beamformer**  
   - Compute **Hybrid beamformer** (weighted combination of ML and MVDR)  
6. **Hybrid Optimization:** Find the best alpha to maximize output SNR.  
7. **SNR Evaluation:** Compare Input, ML, MVDR, and Hybrid SNR values.  
8. **Plot Results:** Save DOA estimation, beam patterns, and SNR comparison in `plots/`.  





