# Student Habits & Academic Performance — K-Means Clustering

Unsupervised machine learning project that identifies distinct student behavior profiles and their relationship to exam performance.

---

## Overview

Can student lifestyle habits predict academic outcomes? This project explores that question on a dataset of 1,000 students, applying K-Means clustering to uncover natural groupings based on study habits, screen time, sleep, mental health, and more.

The analysis runs two experiments — one that includes gender as a feature and one that excludes it — to investigate how gender affects the cluster structure.

---

## Dataset

**File:** `student_habits_performance.csv` — 1,000 student records, 16 features

| Feature | Type | Description |
|---|---|---|
| `study_hours_per_day` | numeric | Average daily study time (hours) |
| `social_media_hours` | numeric | Average daily social media time (hours) |
| `netflix_hours` | numeric | Average daily streaming time (hours) |
| `attendance_percentage` | numeric | Class attendance (0–100%) |
| `sleep_hours` | numeric | Average nightly sleep (hours) |
| `exercise_frequency` | numeric | Exercise sessions per week (0–6) |
| `mental_health_rating` | numeric | Self-rated mental health (1–10) |
| `exam_score` | numeric | Final exam score (0–100) |
| `age` | numeric | Student age (17–24) |
| `gender` | categorical | Female / Male / Other |
| `part_time_job` | categorical | Yes / No |
| `diet_quality` | categorical | Poor / Fair / Good |
| `internet_quality` | categorical | Poor / Average / Good |
| `parental_education_level` | categorical | High School / Bachelor / Master |
| `extracurricular_participation` | categorical | Yes / No |

**Key stats:** avg. exam score 69.6, avg. study hours 3.6/day, 9% missing values in `parental_education_level`

---

## Key Findings

**Strongest predictor of exam score:**

| Feature | Correlation with exam score |
|---|---|
| `study_hours_per_day` | **+0.83** (strong positive) |
| `mental_health_rating` | +0.32 (moderate positive) |
| `social_media_hours` | −0.17 (weak negative) |
| `netflix_hours` | −0.17 (weak negative) |

Study time dominates. Mental health matters. Screen time has a measurable but smaller drag on performance.

---

## Analysis Workflow

### 1. Exploratory Data Analysis
- Distributions and histograms for all numeric features
- Correlation heatmap (filtered for signal above ±0.05 and near-zero relationships)
- Identified right skew in social media and Netflix hours; left skew in attendance and exam scores

### 2. Feature Engineering

| Variable | Encoding | Rationale |
|---|---|---|
| `part_time_job`, `extracurricular_participation` | Binary (0/1) | Only two categories — no need for one-hot |
| `diet_quality`, `internet_quality`, `parental_education_level` | Ordinal (0/1/2) | Natural ordering exists (Poor < Fair/Average < Good) |
| `gender` | One-hot (3 columns) | No ordering; 3 categories |

**Missing data:** `parental_education_level` had 91 missing values (9%). Imputed with the mode (`High School`) — simple and appropriate for MCAR data at this scale.

### 3. Feature Scaling
`StandardScaler` applied to all features before clustering. K-Means uses Euclidean distance, so unscaled features with large ranges (like `attendance_percentage`) would otherwise dominate the results.

### 4. Optimal Cluster Selection — Elbow Method
Inertia (sum of squared distances to cluster centroids) plotted for k = 1–10. The rate of decrease flattens around:
- **k = 4** when gender is included
- **k = 5** when gender is excluded

### 5. K-Means Clustering — Two Experiments

**Experiment A — With gender (4 clusters):**

| Cluster | Size | Profile |
|---|---|---|
| 1 | 241 | Mid-range study hours, mixed habits, average exam scores |
| 2 | 477 | Largest group — moderate habits across all features |
| 3 | 42 | Smallest group — closely mirrors the "Other" gender count, suggesting gender drove this split |
| 4 | 240 | Higher study hours, better exam outcomes |

**Experiment B — Without gender (5 clusters):**

| Cluster | Size | Profile |
|---|---|---|
| 1 | 212 | Higher study hours, strong exam performance |
| 2 | 174 | Lowest study hours, highest Netflix consumption, lowest sleep — lowest exam scores |
| 3 | 194 | High social media usage, lowest exercise, low mental health — below-average exam scores |
| 4 | 196 | Balanced habits, moderate-to-good exam scores |
| 5 | 224 | Average across most features, moderate exam scores |

> Removing gender shifted the elbow from 4 to 5 clusters and produced more evenly sized, habit-driven groups — suggesting gender was structuring the clusters around demographics rather than behavior.

---

## Tech Stack

- **Python 3** — NumPy, Pandas
- **Visualization** — Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (`KMeans`, `StandardScaler`)
- **Environment** — Jupyter Notebook

---

## Setup & Usage

**Prerequisites:** Python 3.8+, Jupyter Notebook or JupyterLab

**1. Clone the repository**
```bash
git clone https://github.com/SvetlanaNMik/KMeans_student_habits_performance.git
cd KMeans_student_habits_performance
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook habits_vs_performance.ipynb
```

**4. Run all cells** — the dataset `student_habits_performance.csv` must be in the same directory as the notebook.

---

## Project Structure

```
├── habits_vs_performance.ipynb   # Full analysis notebook
├── student_habits_performance.csv  # Dataset (1,000 records)
└── README.md
```

---

## Skills Demonstrated

- Data cleaning, imputation, and feature selection
- Exploratory data analysis and correlation analysis
- Categorical encoding strategies (binary, ordinal, one-hot)
- Feature scaling for distance-based algorithms
- Optimal cluster selection using the elbow method
- K-Means clustering and result interpretation
- Comparative experiment design (with vs. without a feature)
