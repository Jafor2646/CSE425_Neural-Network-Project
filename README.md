CSE425 Neural Network Project: K-Means Clustering Implementation
This repository contains the implementation of various approaches to the K-Means clustering algorithm as part of the CSE425 Neural Network course at BRAC University. The project uses the AG News dataset, a text classification dataset, to explore clustering techniques.

The experiments were conducted and executed using Kaggle's computational environment.

Table of Contents
Dataset
Approaches
1. KMeans Without Preprocessing
2. DEC with Preprocessing
3. DEC with Quartile Initialization
How to Use
Results
Acknowledgments
Dataset
The AG News dataset is used for this project. It is a comprehensive text classification dataset consisting of news articles categorized into four classes:

World
Sports
Business
Sci/Tech
For more details about the dataset, visit the HuggingFace AG News dataset page.

Approaches
Three different implementations of the K-Means clustering algorithm are explored in this project:

1. KMeans Without Preprocessing (kmeans_without_preprocessing.py)
This approach directly applies the K-Means algorithm to the raw AG News dataset without any preprocessing steps. It serves as a baseline to understand how the clustering algorithm performs on unprocessed textual data.

2. DEC with Preprocessing (DEC_kmeans.py)
In this approach, the dataset undergoes preprocessing before applying the K-Means algorithm. Additionally, Deep Embedded Clustering (DEC) encoding is utilized to improve the clustering performance. DEC helps reduce dimensionality and extract meaningful features from the data.

3. DEC with Quartile Initialization (DEC_with_quartile_initialization.py)
This is a novel approach introduced in the project. Here, K-Means is combined with DEC encoding, and the initial k cluster centroids are chosen using a quartile-based initialization method. This method aims to improve the quality of the initial centroids, potentially leading to better clustering results.

How to Use
Clone the repository:

bash
git clone https://github.com/Jafor2646/CSE425_Neural-Network-Project.git
cd CSE425_Neural-Network-Project
Install the required dependencies:

bash
pip install -r requirements.txt
Choose one of the approaches to run:

For K-Means without preprocessing:
bash
python kmeans_without_preprocessing.py
For DEC with preprocessing:
bash
python DEC_kmeans.py
For DEC with quartile initialization:
bash
python DEC_with_quartile_initialization.py
Modify the dataset path in the scripts if necessary to ensure the AG News dataset is properly loaded.

Results
The results and performance of each approach will be documented in future updates. The repository includes logs and outputs for each method, which can be analyzed to compare the effectiveness of the different approaches.

Acknowledgments
This project was developed as part of the CSE425 Neural Network course at BRAC University.
The AG News dataset by HuggingFace was instrumental in this project.
Special thanks to Kaggle for providing the computational environment to execute the experiments.
Feel free to contribute to this project by submitting issues or pull requests!