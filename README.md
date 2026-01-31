# 🎓 Skill Gap Awareness System

## Advanced Machine Learning Project - OULAD Analysis

An intelligent learning analytics system that identifies student skill gaps and provides personalized recommendations using **Gaussian Mixture Models (GMM)** for clustering and **Matrix Factorization (MF)** for collaborative filtering.

### 🌐 Live Demo & Quick Access

| Resource | Link |
|----------|------|
| 🚀 **Live Dashboard** | [spiritsfuse-adv-ml-project.streamlit.app](https://spiritsfuse-adv-ml-project.streamlit.app/) |
| 📓 **Open in Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Spiritsfuse/Adv_ML_Project/blob/main/model.ipynb) |
| 📄 **Project Report** | [Project_Report.md](Project_Report.md) |

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Google Colab Setup](#google-colab-setup)
6. [Dependencies](#dependencies)
7. [Model Architecture](#model-architecture)
8. [Key Metrics](#key-metrics)
9. [Acknowledgments](#acknowledgments)

---

## 📖 Project Overview

This project analyzes the **Open University Learning Analytics Dataset (OULAD)** to:

- **Identify student archetypes** based on learning behavior patterns
- **Predict at-risk students** who may need intervention
- **Generate personalized recommendations** to help students improve
- **Provide explainable insights** using "Top Performer Wisdom"

### Problem Statement

Students in online learning environments often struggle to identify their knowledge gaps. This system:

1. Clusters students into behavioral archetypes (High Performer, At-Risk, etc.)
2. Compares individual engagement with successful peer templates
3. Recommends specific learning activities to close skill gaps

---

## ✨ Features

| Feature                                 | Description                                            |
| --------------------------------------- | ------------------------------------------------------ |
| 🎯 **Student Archetype Classification** | GMM clustering into 5 distinct learner profiles        |
| 📊 **Interactive Dashboard**            | Streamlit-based UI for exploration and recommendations |
| 🔮 **Personalized Recommendations**     | Matrix Factorization-based collaborative filtering     |
| 📈 **Success Templates**                | Learning patterns derived from top performers          |
| 💡 **Explainable AI**                   | Human-readable explanations for each recommendation    |

---

## 📊 Dataset

### Source

**OULAD (Open University Learning Analytics Dataset)**

- 22 courses, ~32,000 students
- 10+ million VLE interactions
- Assessment scores, demographics, and outcomes

### Dataset Access

> 📁 **Raw Dataset (~400 MB):** [Google Drive - AML-Project_OULAD_dataset](https://drive.google.com/drive/folders/1A5E-4H31m6Yx3Ld3QXVfR0Mxt1OiV1_2?usp=sharing)

### Dataset Structure

```
dataset/
├── assessments.csv          # Assessment metadata (TMA, CMA, Exam)
├── courses.csv              # Course/module information
├── studentAssessment.csv    # Student scores on assessments
├── studentInfo.csv          # Student demographics & final results
├── studentRegistration.csv  # Registration dates
├── studentVle_0-7.csv       # VLE interaction logs (8 split files, ~55MB each)
└── vle.csv                  # VLE resource types
```

---

## 📁 Project Structure

```
Advanced_ML_Project/
├── 📓 model.ipynb              # MAIN NOTEBOOK - Complete ML pipeline
├── 🖥️ app.py                   # Streamlit Dashboard Application
├── 📋 README.md                # Project documentation
├── 📦 requirements.txt         # Python dependencies
│
├── 📂 dataset/                 # Raw OULAD data (download from Google Drive)
│   └── (14 CSV files)
│
└── 📂 cleaned_data/            # Processed data & model artifacts
    │
    ├── 📂 model_artifacts/     # Trained ML models
    │   ├── gmm_model.pkl              # Gaussian Mixture Model (clustering)
    │   ├── nmf_model.pkl              # Non-negative Matrix Factorization
    │   ├── recommender.pkl            # Base recommender model
    │   ├── enhanced_recommender.pkl   # Enhanced MF with temporal features
    │   ├── temporal_recommender.pkl   # Time-aware recommendations
    │   ├── explainer.pkl              # Explanation generator
    │   ├── feature_scaler.pkl         # Feature normalization scaler
    │   └── dynamic_tracker.pkl        # Dynamic progress tracker
    │
    ├── 📂 cluster_results/     # Clustering outputs
    │   ├── cluster_assignments.csv    # Student → Archetype mappings
    │   ├── archetype_metadata.csv     # Archetype descriptions
    │   ├── archetype_success_rates.csv# Success rates per archetype
    │   └── clustering_features_scaled.csv
    │
    ├── 📂 recommendation_data/ # Recommendation system data
    │   ├── success_templates_v2.csv   # Success behavior patterns
    │   ├── course_templates.json      # Course-specific templates
    │   ├── course_archetype_templates.json
    │   ├── interaction_matrix_full.csv# User-item interactions
    │   ├── activity_success_correlations.csv
    │   ├── activity_latent_factors.csv
    │   ├── student_latent_factors.csv
    │   ├── intervention_strategies.csv
    │   └── phase_template_*.csv       # Phase-wise templates
    │
    └── 📂 feature_data/        # Engineered features
        ├── feature_store.csv          # Complete feature matrix
        ├── student_weekly_features.csv# Weekly engagement features
        ├── student_activity_summary.csv
        ├── unified_students.csv       # Merged student records
        ├── student_assessment_clean.csv
        ├── assessments_clean.csv
        └── vle_reference.csv          # VLE activity reference
```

---

## ☁️ Google Colab Setup

### Step 1: Open the Notebook

Open `model.ipynb` in Google Colab using the provided link or upload directly.

### Step 2: Mount Google Drive & Download Dataset

Run this cell at the beginning of the notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create dataset directory
!mkdir -p dataset

# Copy dataset from Google Drive
# Dataset Link: https://drive.google.com/drive/folders/1A5E-4H31m6Yx3Ld3QXVfR0Mxt1OiV1_2
!cp -r "/content/drive/MyDrive/AML-Project_OULAD_dataset/"* ./dataset/
```

### Step 3: Install Dependencies

```python
!pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit -q
```

### Step 4: Run All Cells

Execute all notebook cells sequentially to:

1. Load and preprocess OULAD data
2. Engineer behavioral features
3. Train GMM clustering model
4. Build Matrix Factorization recommender
5. Generate recommendations and visualizations

### Running the Dashboard in Colab

To run the Streamlit dashboard in Colab:

```python
# Install required packages
!pip install streamlit pyngrok -q

# Write app.py content (already in notebook)
# Then run:
from pyngrok import ngrok

# Start Streamlit in background
!streamlit run app.py --server.port 8501 &>/dev/null &

# Create public URL
public_url = ngrok.connect(8501)
print(f"🌐 Dashboard URL: {public_url}")
```

---

## 📦 Dependencies

| Package      | Purpose                   |
| ------------ | ------------------------- |
| pandas       | Data manipulation         |
| numpy        | Numerical computing       |
| scikit-learn | ML algorithms (GMM, NMF)  |
| scipy        | Scientific computing      |
| matplotlib   | Static visualizations     |
| seaborn      | Statistical plots         |
| plotly       | Interactive charts        |
| streamlit    | Dashboard web application |

Full list available in `requirements.txt`

---

## 🧠 Model Architecture

### Phase 1: Data Preprocessing

- Join 7 OULAD tables into unified student records
- Handle missing values and outliers
- Create temporal features (weekly engagement patterns)

### Phase 2: Feature Engineering

- **Engagement Features**: clicks/week, active days, consistency score
- **Performance Features**: assessment scores, late submissions
- **Behavioral Features**: recency, burstiness, session patterns

### Phase 3: Student Clustering (GMM)

- Gaussian Mixture Model with 5 components
- **Student Archetypes:**
  1. 🌟 **High Performer** - Strong engagement & scores
  2. ⚡ **Talented but Inconsistent** - High potential, irregular patterns
  3. 📚 **Moderate Performer** - Average across metrics
  4. 🔧 **Early Struggler** - Needs early intervention
  5. ⚠️ **Disengaged At-Risk** - Critical attention required

### Phase 4: Collaborative Filtering (MF)

- Non-negative Matrix Factorization (NMF)
- User-Item interaction matrix from VLE clicks
- Top Performer Wisdom for recommendation reranking

### Phase 5: Recommendation Engine

- Skill gap = Success Template - Current Engagement
- Priority scoring based on:
  - Gap magnitude
  - Activity-success correlation
  - Peer similarity

---

## 📊 Key Metrics

| Metric                      | Value   |
| --------------------------- | ------- |
| Total Students Analyzed     | ~32,000 |
| VLE Interactions Processed  | ~10M    |
| Clustering Silhouette Score | 0.45    |
| Recommendation Coverage     | 95%+    |

---

## 🙏 Acknowledgments

- **Dataset**: [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
- **Libraries**: scikit-learn, Streamlit, Plotly

---

## 📝 License

This project is for educational purposes as part of the Advanced Machine Learning course.

---

## 👥 Team

**Team Leader**: Dhruv Sharma  
**Team Members**: Yashwardhan Singh | Kartavya Panchal | Ojas Maheshwari | Tushar Shaw

---

_Last Updated: January 2026_
