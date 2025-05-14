🧠 CSE425 Neural Network Project: K-Means Clustering on AG News
This repository contains the implementation of various K-Means clustering techniques developed as part of the CSE425 Neural Network course at BRAC University. The project applies different clustering strategies to the AG News dataset to analyze and improve the effectiveness of unsupervised text classification.

🧪 All experiments were conducted on Kaggle's cloud-based computational environment.

📑 Table of Contents
📚 Dataset

🧭 Approaches

1. K-Means Without Preprocessing

2. DEC with Preprocessing

3. DEC with Quartile Initialization

🚀 How to Use

📊 Results

🙏 Acknowledgments

📚 Dataset
We use the AG News dataset, a popular text classification benchmark consisting of news articles categorized into four distinct classes:

🌍 World

🏅 Sports

💼 Business

🔬 Sci/Tech

This dataset enables a comprehensive exploration of clustering methods in the context of textual data.

🧭 Approaches
Three different implementations of the K-Means clustering algorithm are explored in this project:

1. K-Means Without Preprocessing [kmeans_without_preprocessing.py]
This baseline method applies K-Means directly on the raw AG News text data without any preprocessing. It helps demonstrate how clustering performs without feature engineering or dimensionality reduction.

2. Deep Embedded Clustering (DEC) with Preprocessing [DEC_kmeans.py]
In this approach:

The data undergoes preprocessing and embedding using Sentence-BERT.

A Deep Autoencoder is trained to reduce dimensionality.

K-Means is applied on the encoded features to form clusters.

DEC enhances the clustering performance by extracting dense, informative representations of the textual data.

3. DEC with Quartile Initialization [DEC_with_quartile_initialization.py]
This is a novel approach introduced in the project. It extends the DEC method with a custom quartile-based initialization for K-Means centroids:

Initial centroids are chosen using interquartile statistics, aiming to better represent the distribution of embeddings.

This often leads to faster convergence and improved cluster quality.

🚀 How to Use
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/Jafor2646/CSE425_Neural-Network-Project.git
cd CSE425_Neural-Network-Project
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run a Clustering Approach:

K-Means Without Preprocessing

bash
Copy
Edit
python kmeans_without_preprocessing.py
DEC with Preprocessing

bash
Copy
Edit
python DEC_kmeans.py
DEC with Quartile Initialization

bash
Copy
Edit
python DEC_with_quartile_initialization.py
📁 Ensure the AG News dataset is correctly loaded. Modify dataset paths if needed in the scripts.

📊 Results
Performance metrics such as Silhouette Score and Davies-Bouldin Index are used to evaluate clustering quality. Logs and outputs are available in the repository for each method.

📝 Detailed performance comparison will be added in future updates.

🙏 Acknowledgments
This project was developed as part of the CSE425 Neural Network course at BRAC University.

Thanks to HuggingFace for the AG News dataset.

Special appreciation to Kaggle for providing the cloud environment to execute experiments.

🤝 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

🧾 License
This project is open-source and available under the MIT License.