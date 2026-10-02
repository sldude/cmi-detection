# CMI Detection: Predicting Body-Focused Repetitive Behaviors from Wrist Sensor Data

Final project for **Deep Neural Networks (605.742), Johns Hopkins University**, entered into the Kaggle competition [CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).

**Team 5:** Arlene Keith, Bruno Mfuh, Steven Liu

Presentation video: https://youtu.be/v8ZtbRfWTxg

---

## Overview

Body-Focused Repetitive Behaviors (BFRBs) such as hair pulling, skin picking, and scratching are self-directed habits that can cause physical harm and are commonly associated with anxiety and OCD. The Child Mind Institute built **Helios**, a wrist-worn device, to detect these behaviors automatically. The goal of the competition is to classify which gesture a participant performed, given time-series data from the device.

Helios records three sensor types:

| Sensor | Columns | Description |
|---|---|---|
| IMU | `acc_[x/y/z]`, `rot_[w/x/y/z]` | Linear acceleration (m/s²) and orientation quaternion |
| Thermopiles | `thm_1` … `thm_5` | Temperature (°C) from 5 sensors, captures body heat |
| Time-of-Flight | `tof_[1-5]_v[0-63]` | Five 8×8 proximity grids (0–254, −1 = no response) |

Each sequence moves through three phases: **Transition** (moving from rest to the target area), **Pause** (brief inactivity), and **Gesture** (performing the BFRB-like or non-BFRB-like motion).

### Gestures

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body positions: sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side. The gestures are listed below.

| BFRB-Like Gesture (Target Gesture) |
|---|
| Above ear - Pull hair |
| Forehead - Pull hairline |
| Forehead - Scratch |
| Eyebrow - Pull hair |
| Eyelash - Pull hair |
| Neck - Pinch skin |
| Neck - Scratch |
| Cheek - Pinch skin |

| Non-BFRB-Like Gesture (Non-Target Gesture) |
|---|
| Drink from bottle/cup |
| Glasses on/off |
| Pull air toward your face |
| Pinch knee/leg skin |
| Scratch knee/leg skin |
| Write name on leg |
| Text on phone |
| Feel around in tray and pull out an object |
| Write name in air |
| Wave hello |

### Dataset

- **Training set:** 8,151 sequences of varying length (575,052 rows × 349 columns, 1.12 GB total), plus per-subject demographic data (age, sex, handedness, height, shoulder-to-wrist and elbow-to-wrist lengths).
- **Hidden test set:** ~3,500 sequences, half of which contain **IMU data only**, to measure whether thermopile and ToF sensors improve gesture identification.
- Kaggle runs the submission notebook against the hidden test set through an evaluation API that serves one sequence at a time.

### Scoring metric

The average of a **binary F1** (BFRB vs. non-BFRB) and a **macro F1** across all gesture classes. This balances detecting whether a target behavior occurred with identifying exactly which gesture it was.

---

## Final Model: Dual-Network Ensemble

Our final solution is an ensemble of **two independent deep feedforward networks** trained on sequence-level engineered features:

- a **Main Model** that uses all sensors (IMU, thermopiles, ToF) plus demographics, and
- an **IMU Model** that uses motion features only and acts as a fallback when ToF/thermopile data is missing or unreliable.

Both models process each input simultaneously, and their probability outputs are combined with weights that adapt to a real-time **sensor-quality** assessment:

```
Final Prediction = α × Main_Prediction + β × IMU_Prediction,   where α + β = 1.0
```

When ToF sensors are working, the Main Model dominates; when they fail or are missing (as in the IMU-only half of the test set), weight shifts toward the IMU Model.

### Preprocessing pipeline

```
Raw sequence data
  → 1. Missing data imputation
  → 2. Sequence-level aggregated feature extraction
  → 3. Merge with demographic data
  → 4. Feature engineering
  → 5. Outlier handling & weak feature removal
  → 6. Feature transformation / scaling
  → Model training / prediction
```

1. **Missing data imputation.** IMU-only sequences and known sensor communication failures leave NaNs in thermopile and ToF columns. These are imputed with training-set medians.
2. **Sequence-level aggregation.** Because the time series is noisy, each sequence is summarized to capture broader gesture-specific trends:
   - IMU: mean, standard deviation, min, and max per channel (e.g., `rot_x_mean`, `rot_x_std`)
   - Thermopiles: mean and std per sensor, plus `temp_range` and `temp_gradient`
   - ToF: mean and std per sensor, plus `tof_i_valid_ratio` (valid readings / total readings)
3. **Demographic merge.** Aggregated features are joined with demographic data by subject; missing demographic values are imputed with the mode.
4. **Feature engineering:**
   - *Thermal interaction:* `temp_diff_1_4`, `temp_ratio_1_4`, `temp_variance_ratio`, capturing spatial heat patterns that indicate hand proximity to the face
   - *Cross-modal:* `movement_heat_coupling` (acceleration magnitude × temperature range), `movement_temp_sync` (average acceleration std / temperature gradient)
   - *Rotation:* `hand_elevation` = arctan(rot_y_mean / rot_z_mean), `wrist_twist` = arctan(rot_x_mean / rot_w_mean), `orientation_stability` = 1 / ‖rot std‖
   - *ToF interaction:* `tof_center_activity`, `tof_center_variance`, `tof_reliability_score` over sensors 1–4
   - *Dynamics:* `gesture_intensity` (acceleration magnitude), `gesture_smoothness` (1 / jerk), `movement_coordination`
5. **Outlier handling and weak feature removal.** Numeric features are clipped to the 5th–95th percentile range. Weak or redundant features (`thm_5_mean`, `thm_5_std`, ToF sensor 5 aggregates, and `rot_z_std`) are dropped.
6. **Transformation and scaling.** A Yeo-Johnson power transform followed by `StandardScaler`, both fit on training data and reapplied at test time. The `gesture` target is encoded with `LabelEncoder`.

### Handling class imbalance

The training data has a **~4:1 imbalance** between majority classes (~640 sequences) and minority classes (~161 sequences), with the minority classes mostly non-BFRB gestures such as *Drink*, *Glasses on/off*, *Pinch knee*, and *Scratch knee*. We addressed this in two ways.

**Inverse-frequency class weights.** Minority classes receive a weight of ~3.97 versus ~1.00 for majority classes, giving underrepresented gestures roughly 4× more importance.

**Adaptive data augmentation** with per-class multipliers based on difficulty:

| Multiplier | Classes |
|---|---|
| 3.0× | Scratch knee/leg skin |
| 2.5× | Eyebrow - pull hair, Neck - pinch skin, Neck - scratch |
| 2.0× | Pinch knee/leg skin |
| 1.5–1.8× | Remaining imbalanced classes |

Each augmented sample is created with one of three techniques:

- **Gaussian noise (40%)** with σ = 0.02–0.03 (higher for problem classes) to simulate sensor variability
- **Scaling (30%)** by a uniform factor in [0.85, 1.15] to account for movement-intensity variation
- **Within-class mixup (30%)** with a Beta(0.2, 0.2) mixing parameter to smooth decision boundaries

Augmentation targets 1.8× the median class size and is capped at 3× per class to prevent over-augmentation, growing the training set from **8,151 to 14,423 samples**.

### Main Model

| | |
|---|---|
| Input | 120 features from all sensors |
| Architecture | Dense(256) → Attention Gate(256) → Dense(128) → Dense(64) + Residual → Dense(32) → Softmax(18) |
| Parameters | ~85,000 |
| Attention | Sigmoid gating at 256 units for dynamic feature focusing |
| Residual | Skip connection from the input to the 64-unit layer |
| Regularization | BatchNormalization after each dense layer; progressive dropout 0.3 → 0.15 |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999) |
| Learning rate | One-cycle schedule (0.0002 → 0.006 → 0) |
| Loss | Focal loss (γ = 1.5, α = 0.3) to address class imbalance |
| Batch size / epochs | 128 / 50 with early stopping |

**Feature groups:** 49 IMU features (accelerometer and quaternion statistics, jerk, smoothness), 60 ToF features (sampled pixels, missing-rate indicators, proximity scores), 12 temperature features, and 7 demographic features.

**Training progression:** accuracy rose from 22.8% at epoch 1 to 71.4% at epoch 10, 83.2% at epoch 30, and converged at about 87.4% by epoch 50.

### IMU Model

| | |
|---|---|
| Input | 49 motion + demographic features |
| Architecture | Dense(128) → Attention Gate(128) → Dense(64) → Dense(32) + Residual → Dense(16) → Softmax(18) |
| Parameters | ~35,000 |
| Attention | Smaller 128-unit sigmoid gate |
| Residual | Skip connection from the input to the 32-unit layer |
| Regularization | Same progressive dropout strategy as the Main Model |
| Loss | Sparse categorical cross-entropy with class weights |

**Feature groups:** 15 accelerometer features (mean, std, range per axis), 13 rotation/quaternion features, 14 motion features (acceleration magnitude, jerk, angular velocity, rotation magnitude, smoothness, complexity, vertical dominance, lateral movement ratio), and 7 demographic features used for motion scaling.

Sequences range from about 40 to 300 frames (~0.8–6 seconds at 50 Hz) and are processed in their entirety, with no padding or truncation. The IMU Model emphasizes temporal patterns (jerk, smoothness), vertical position relative to gravity, and quaternion-based orientation changes.

---

## Results

### Local validation

| Model | Accuracy | Macro F1 | Binary F1 | Competition score |
|---|---|---|---|---|
| Main Model | 95.76% | 0.966 | 0.9991 | **0.9826** |
| IMU Model | 60.54% | 0.680 | 0.9881 | 0.8339 |
| Ensemble | | | | **0.9920** |

Main Model BFRB detection:

- **Precision:** 99.81% (only 2 false positives)
- **Recall:** 99.43%
- **Mean prediction confidence:** 90.7%
- **0% error rate** on 7 of the 18 gesture classes

The hardest gestures were *Neck - scratch* (18% error), which was most often confused with *Neck - pinch skin*, and *Eyebrow - pull hair* (10.9% error), which was confused with *Eyelash - pull hair*. These pairs involve nearly identical hand motion in adjacent body regions, which the ToF sensors' resolution may not distinguish.

### Kaggle leaderboard

The ensemble achieved a public leaderboard score of **0.70**, compared with 0.60–0.66 for our earlier approaches.

| Course module | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| Public LB score | 0.23 | 0.43 | 0.66 | 0.66 | 0.68 | 0.68 | 0.69 | 0.70 | 0.70 |

Total submissions: 65.

---

## Other Approaches Explored

Before settling on the ensemble, we explored:

- **CNN/LSTM sequence modeling** (`Team_5_Final_CNNDNN_1.ipynb`, `team5-submission-cnn-dnn-2.ipynb`): a BiLSTM that predicts the *Gesture* phase at each timestep, followed by a BiLSTM classifier for IMU/thermopile data and a time-distributed 2D CNN for the 8×8 ToF grids.
- **Sampling strategies** with several neural network architectures, which did not yield substantial improvements.
- **Ensembles of more than two models**, which failed at inference time with our feature engineering pipeline.

---

## Repository Contents

| File | Description |
|---|---|
| `draft-8.ipynb` | Final ensemble notebook: preprocessing, feature engineering, augmentation, Main and IMU models, and Kaggle inference. |
| `Team_5_Final_CNNDNN_1.ipynb` | Exploratory CNN + BiLSTM sequence-modeling pipeline with EDA visualizations. |
| `team5-submission-cnn-dnn-2.ipynb` | Earlier Kaggle submission using the BiLSTM phase-prediction pipeline. |
| `Results Draft-12-V1.rtf` | Training logs and local evaluation output for the final models. |
| `FP Presentation - Team 5-PDF.pdf` | Final project slide deck. |

---

## Running the Notebooks

### Hardware

Models were trained on **Google Colab Pro** (2 CPUs, 1 NVIDIA T4 GPU, 12.7 GB RAM, 15 GB GPU memory, 75 GB storage). Preprocessing exceeds the 12 GB RAM limit of free Colab. Kaggle submissions must complete within 9 hours of runtime with internet access disabled.

### Requirements

```
python >= 3.10
tensorflow >= 2.15
numpy
pandas
polars
scipy
scikit-learn
plotly
matplotlib
kaggle
```

### Getting the data

1. Accept the competition rules on Kaggle.
2. Place your `kaggle.json` API token in `~/.kaggle/`.
3. Download and unzip:

```bash
kaggle competitions download -c cmi-detect-behavior-with-sensor-data
unzip -o cmi-detect-behavior-with-sensor-data.zip
```

Submissions run through Kaggle's `kaggle_evaluation.cmi_inference_server`, so the inference cells only fully work inside a Kaggle notebook. Trained model files are not included in this repository.

---

## Limitations and Future Work

### Modeling issues encountered

- **Large gap between local and Kaggle scores** (0.98–0.99 locally vs. 0.70 on the leaderboard).
- **RAM usage** during feature engineering exceeded free Colab limits.
- **Missing ToF data** and class imbalance made early submissions score poorly.

### Limitations

- **ToF sensor dependency:** performance drops from 98.3% to 83.4% when ToF sensors fail.
- **IMU-only confusion:** high error rates for spatially similar gestures, such as *Neck - scratch* vs. *Above ear - pull hair*.
- **Augmentation limits:** weights and augmentation cannot fix motion-similarity issues, returns diminished above 2.5×, and the 3.0× multiplier risks over-augmenting *Scratch knee/leg skin*.
- **Feature engineering bias:** hand-crafted aggregate features may miss subtle temporal patterns.

### Future work

- **Temporal modeling:** LSTM/GRU models on raw sequences without aggregation, and hierarchical models for each gesture phase (approach, contact, retract)
- **Augmentation:** GAN-based synthetic gesture sequences and physics-based biomechanical simulation
- **Research directions:** self-supervised pretraining on unlabeled gesture data and reinforcement learning for intervention strategies

---

## References

1. CMI - Detect Behavior with Sensor Data. Kaggle, 2025. https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/overview
2. Garey, J. (2025). *What Is Excoriation, or Skin-Picking?* Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/
3. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly Media.
4. *Gesture Recognition Using Inertial Measurement Units.* MathWorks. https://www.mathworks.com/help/nav/ug/gesture-recognition-using-inertial-measurement-units.html
5. Abdalla, H. E. A. (2021). *Hand Gesture Recognition Based on Time-of-Flight Sensors.* Master's thesis, Politecnico di Torino. https://webthesis.biblio.polito.it/19162/1/tesi.pdf
6. Lee, S. (2025). *Mastering Yeo-Johnson Transformation.* Number Analytics. https://www.numberanalytics.com/blog/yeo-johnson-data-normalization
7. Martinelli, K. (2025). *What is Trichotillomania?* Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/
8. Panarin, R. (2024). *Basic Data Augmentation Method Applied to Time Series.* Mad Devs. https://maddevs.io/writeups/basic-data-augmentation-method-applied-to-time-series/
9. Tseng, Y.-H., & Wen, C.-Y. (2023). Hybrid Learning Models for IMU-Based HAR with Feature Analysis and Data Correction. *Sensors*, 23(18), 7802. https://doi.org/10.3390/s23187802
10. Zhang, H., Cisse, M., Dauphin, Y., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. https://arxiv.org/abs/1710.09412
