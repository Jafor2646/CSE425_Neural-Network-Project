# 🧠 CSE425 Neural Network Project: K-Means Clustering on AG News 📰

**Exploring Unsupervised Text Classification with Diverse K-Means Strategies**

This repository showcases the implementation of various K-Means clustering techniques, developed as part of the CSE425 Neural Network course at BRAC University. The project applies different clustering strategies to the AG News dataset to analyze and enhance the effectiveness of unsupervised text classification.

🧪 *All experiments were conducted on Kaggle's cloud-based computational environment.*

## 📑 Table of Contents
- [Dataset](#-dataset)
- [✨ Key Features](#-key-features)
- [🧭 Explored Approaches](#-explored-approaches)
  - [1. K-Means Without Preprocessing](#1-k-means-without-preprocessing)
  - [2. Deep Embedded Clustering (DEC) with Preprocessing](#2-deep-embedded-clustering-dec-with-preprocessing)
  - [3. DEC with Quartile Initialization](#3-dec-with-quartile-initialization)
- [🚀 How to Use](#-how-to-use)
- [📊 Results](#-results)
- [🙏 Acknowledgments](#-acknowledgments)
- [🤝 Contributions](#-contributions)
- [🧾 License](#-license)

## 📚 Dataset

We utilize the **AG News dataset**, a widely-used benchmark for text classification. It comprises news articles categorized into four distinct classes:

- 🌍 **World**
- 🏅 **Sports**
- 💼 **Business**
- 🔬 **Sci/Tech**

This dataset provides a rich environment for exploring clustering methods on textual data.

## ✨ Key Features

- **Multiple K-Means Implementations**: Explore baseline, preprocessed, and advanced DEC-based clustering.
- **AG News Dataset**: Leverages a standard dataset for robust text clustering analysis.
- **Deep Embedded Clustering (DEC)**: Implements DEC for learning meaningful representations.
- **Novel Quartile Initialization**: Introduces a custom K-Means centroid initialization technique.
- **Unsupervised Learning Focus**: Concentrates on text classification without labeled data.

## 🧭 Explored Approaches

This project investigates three distinct implementations of the K-Means clustering algorithm:

### 1. K-Means Without Preprocessing
   - 📜 **Script:** `kmeans_without_preprocessing.py`
   - **Description:** This baseline method applies K-Means directly to the raw AG News text data without any preprocessing. It serves as a benchmark to understand clustering performance without feature engineering or dimensionality reduction.

### 2. Deep Embedded Clustering (DEC) with Preprocessing
   - 📜 **Script:** `DEC_kmeans.py`
   - **Description:** In this approach:
     - Data undergoes preprocessing and embedding using **Sentence-BERT**.
     - A **Deep Autoencoder** is trained to reduce dimensionality.
     - K-Means is applied to the encoded features to form clusters.
   - **Benefit:** DEC aims to enhance clustering performance by extracting dense, informative representations from the textual data.

### 3. DEC with Quartile Initialization
   - 📜 **Script:** `DEC_with_quartile_initialization.py`
   - **Description:** This novel approach, introduced in this project, extends the DEC method with a custom quartile-based initialization for K-Means centroids:
     - Initial centroids are chosen using **interquartile statistics**, aiming for a better representation of the embedding distribution.
   - **Benefit:** This method often leads to faster convergence and improved cluster quality.

## 🚀 How to Use

Follow these steps to get the project running:

1.  **Clone the Repository:**
    ````bash
    git clone https://github.com/Jafor2646/CSE425_Neural-Network-Project.git
    cd CSE425_Neural-Network-Project
    ````

2.  **Install Dependencies:**
    ````bash
    pip install -r requirements.txt
    ````
    *Ensure you have Python and pip installed.*

3.  **Run a Clustering Approach:**
    Choose one of the scripts to execute:

    *   **K-Means Without Preprocessing:**
        ````bash
        python kmeans_without_preprocessing.py
        ````
    *   **DEC with Preprocessing:**
        ````bash
        python DEC_kmeans.py
        ````
    *   **DEC with Quartile Initialization:**
        ````bash
        python DEC_with_quartile_initialization.py
        ````

4.  **Dataset Note:**
    📁 Ensure the AG News dataset is correctly loaded. You might need to modify dataset paths within the scripts if your setup differs.

## 📊 Results

Clustering quality is evaluated using performance metrics such as the **Silhouette Score** and **Davies-Bouldin Index**.
Detailed logs and outputs for each method are available within the repository.

📝 *A comprehensive performance comparison and visualizations will be added in future updates.*

## 🙏 Acknowledgments

- This project was developed as part of the **CSE425 Neural Network** course at BRAC University.
- Thanks to **HuggingFace** for providing the AG News dataset.
- Special appreciation to **Kaggle** for the cloud environment used for experiments.

## 🤝 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests to help improve this project.

## 🧾 License

This project is open-source and available under the **MIT License**.