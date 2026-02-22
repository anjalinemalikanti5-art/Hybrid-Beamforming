# Hybrid Machine Learning and Optimization based Spatial Filtering Framework

**Enhancing Signal-to-Noise Ratio (SNR) in antenna arrays using ML, MVDR, and Hybrid Beamforming.**

---

## Description

This project implements a hybrid beamforming framework combining:

- **Machine Learning (Linear Regression for DOA estimation)**  
- **MVDR beamforming**  
- **Hybrid optimization**  

The goal is **robust sensor array signal processing** using a dataset of antenna snapshots.

---

## Features

- DOA estimation (Noisy, Denoised, ML Prediction)  
- SNR improvement (Input vs ML vs MVDR vs Hybrid)  
- Beam patterns (ML, MVDR, Hybrid)  
- Before/After beamforming signal comparison  
- 3D hybrid beam pattern visualization  
- SNR vs alpha sweep plot  

---

## Dataset

Place your CSV dataset file inside the `dataset/` folder.  
**Current example:** `C_44_train_converted.csv`

---

## Project Overview

Hybrid Beamforming is a signal processing technique used to improve antenna array performance by combining multiple beamforming approaches. This project focuses on **Direction of Arrival (DOA) estimation** and **SNR enhancement** using:

- ML-based DOA prediction  
- MVDR (Minimum Variance Distortionless Response) beamforming  
- Hybrid optimization of beamforming weights  

**Key objectives:**

1. Load and process antenna snapshot data  
2. Denoise noisy DOA measurements  
3. Predict DOA using a linear ML model  
4. Compute covariance matrices and derive beamforming weights  
5. Optimize hybrid beamforming to maximize SNR  
6. Visualize results through plots for DOA estimation, beam patterns, and SNR comparison  

---

## Project Workflow

### Numbered Steps

1. **Data Input:** Load `C_44_train_converted.csv` from the `dataset/` folder  
2. **DOA Denoising:** Apply Savitzky–Golay filter to smooth noisy angles  
3. **ML Prediction:** Perform linear regression to estimate DOA per snapshot  
4. **Covariance Calculation:** Compute covariance matrix of received signals  
5. **Beamforming:**  
   - Compute ML beamformer  
   - Compute MVDR beamformer  
   - Compute Hybrid beamformer (weighted combination of ML and MVDR)  
6. **Hybrid Optimization:** Find the best alpha to maximize output SNR  
7. **SNR Evaluation:** Compare Input, ML, MVDR, and Hybrid SNR values  
8. **Plot Results:** Save DOA estimation, beam patterns, and SNR comparison in `plots/`  

---

### Workflow Diagram

 Data Input -> DOA Denoising -> ML Prediction -> Covariance  -> Beamforming -> Hybrid Opt. -> SNR Evaluation -> Plot Results

### Running Instructions
 
## 1️⃣ C++ Code

cd code

g++ hybrid.cpp -o hybrid -std=c++17 -I /path/to/eigen
./hybrid

Ensure C_44_train_converted.csv is in the dataset/ folder.

## 2️⃣ Python Notebook

jupyter notebook notebooks/hybrid_beamforming.ipynb

Run all cells to process data, apply DOA denoising, compute ML/MVDR/Hybrid beamformers, optimize hybrid weights, and generate plots in plots/.

### Results / Plots

DOA Estimation: Shows noisy, denoised, and ML predicted angles

Beam Patterns: ML, MVDR, and Hybrid beam responses

SNR Comparison: Input vs ML vs MVDR vs Hybrid

