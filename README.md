# Student Habits & Academic Performance - K-Means Clustering

Unsupervised machine learning project that identifies distinct student behavior profiles and their relationship to exam performance.

---

## Overview

Can student lifestyle habits predict academic outcomes? This project explores that question on a dataset of 1,000 students, applying K-Means clustering to uncover natural groupings based on study habits, screen time, sleep, mental health, and more.

The analysis runs two back-to-back experiments - one that includes gender as a feature and one that excludes it - to investigate how a demographic variable affects cluster structure.

---

## Dataset

**File:** `student_habits_performance.csv` - 1,000 student records, 16 features

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

**Strongest predictors of exam score:**

| Feature | Correlation with exam score |
|---|---|
| `study_hours_per_day` | **+0.83** (strong positive) |
| `mental_health_rating` | +0.32 (moderate positive) |
| `social_media_hours` | −0.17 (weak negative) |
| `netflix_hours` | −0.17 (weak negative) |

**Gender inclusion finding:** When gender is included as a feature, it completely dominates cluster structure. At k=2, K-Means separates Male (477 students) from Female+Other (523 students) - with no meaningful habit or performance differences between the two groups. Study hours, attendance, and exam scores are nearly identical across both clusters (silhouette: 0.114).

**Performance archetypes (without gender, k=3):** Removing gender reveals three habit-driven performance tiers:

| Cluster | Size | Study h/day | Mental health | Exam score | Defining trait |
|---|---|---|---|---|---|
| High Performers | 415 | **4.50** | **6.50** | **82.80** | High study hours, best mental health, almost no part-time jobs |
| Struggling Students | 374 | **2.58** | **4.24** | **55.77** | Low study hours, poorest mental health, highest social media use |
| Working Students | 211 | 3.41 | 5.48 | 68.16 | **100% hold part-time jobs**, lowest attendance |

Study time and mental health are the two features that cleanly separate high performers from struggling students. Working students form a structurally distinct group regardless of their performance level.

---

## Analysis Workflow

### 1. Exploratory Data Analysis
- Distributions and histograms for all numeric features
- Dual correlation heatmap: one filtered for |r| > 0.05 (meaningful signal), one for |r| ≤ 0.01 (near-zero relationships)
- Identified right skew in social media and Netflix hours; left skew in attendance and exam scores

### 2. Feature Engineering

| Variable | Encoding | Rationale |
|---|---|---|
| `part_time_job`, `extracurricular_participation` | Binary (0/1) | Only two categories - no need for one-hot |
| `diet_quality`, `internet_quality`, `parental_education_level` | Ordinal (0/1/2) | Natural ordering exists (Poor < Fair/Average < Good) |
| `gender` | One-hot (3 columns) | No ordering; 3 categories |

**Missing data:** `parental_education_level` had 91 missing values (9%). Imputed with the mode (`High School`) - simple and appropriate for MCAR data at this scale.

### 3. Feature Scaling
`StandardScaler` applied to all features before clustering. K-Means uses Euclidean distance, so unscaled features with large ranges would otherwise dominate the results.

### 4. Optimal Cluster Selection - Elbow Method
Inertia (sum of squared distances to cluster centroids) plotted for k = 1–10. The rate of decrease flattens around:
- **k = 2** when gender is included
- **k = 3** when gender is excluded

### 5. K-Means Clustering - Two Experiments

**Experiment A - With gender (k=2):**

| Cluster | Size | Profile |
|---|---|---|
| 1 | 477 | 100% Male - habits and exam scores nearly identical to Cluster 2 |
| 2 | 523 | ~92% Female + ~8% Other - gender drives the split entirely |

At k=2, the algorithm uses both cluster slots to separate Male from Female+Other. No behavioral signal is present.

**Experiment B - Without gender (k=3):**

| Cluster | Size | Profile |
|---|---|---|
| 1 | 415 | High performers - highest study hours (4.50 h/day), best mental health (6.50), highest exam scores (82.80) |
| 2 | 374 | Struggling students - lowest study hours (2.58 h/day), poorest mental health (4.24), lowest exam scores (55.77) |
| 3 | 211 | Working students - 100% hold part-time jobs, moderate performance, lowest attendance |

> Removing gender shifted the elbow from k=2 to k=3 and produced performance-driven groups defined primarily by study time and mental health - confirming that gender was masking the behavioral structure of the data.

---

## Tech Stack

- **Python 3** - NumPy, Pandas
- **Visualization** - Matplotlib, Seaborn
- **Machine Learning** - Scikit-learn (`KMeans`, `StandardScaler`, `silhouette_score`)
- **Environment** - Jupyter Notebook

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

**4. Run all cells** - the dataset `student_habits_performance.csv` must be in the same directory as the notebook.

---

## Project Structure

```
├── habits_vs_performance.ipynb     # Full analysis notebook
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
- Comparative experiment design (with vs. without a demographic feature)
- Cluster quality validation with silhouette score
- Centroid interpretation via inverse-transform scaling
