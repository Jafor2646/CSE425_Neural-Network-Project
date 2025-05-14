#Install dependencies
!pip install -U fsspec huggingface_hub sentence-transformers scikit-learn matplotlib


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

#Load and clean dataset
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet'
}
df = pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["train"])
df = df[df['text'].str.len() > 50]  # Filter short texts
texts = df['text'].tolist()

#Generate embeddings
embedder = SentenceTransformer('paraphrase-MiniLM-L12-v2')
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
X_scaled = StandardScaler().fit_transform(embeddings)
X = torch.tensor(X_scaled, dtype=torch.float32)

#Define Autoencoder
class TextAutoencoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32):
        super(TextAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

#Pretrain autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextAutoencoder(input_dim=384, latent_dim=32).to(device)

pretrain_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Pretraining Autoencoder...")
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in pretrain_loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, _ = model(x_batch)
        loss = criterion(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#Clustering Layer
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim):
        super(ClusteringLayer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2)
        q = 1.0 / (1.0 + dist)
        q = q ** ((1 + 1) / 2)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

#Initialize Clustering Layer with KMeans
model.eval()
with torch.no_grad():
    _, latent_init = model(X.to(device))
latent_np = latent_init.cpu().numpy()
kmeans = KMeans(n_clusters=4, n_init=20).fit(latent_np)

clustering_layer = ClusteringLayer(n_clusters=4, latent_dim=32).to(device)
clustering_layer.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

#Train DEC
dec_optimizer = optim.Adam(list(model.parameters()) + list(clustering_layer.parameters()), lr=0.0005)
kl_loss = nn.KLDivLoss(reduction='batchmean')

print("Training DEC...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in pretrain_loader:
        x_batch = batch[0].to(device)
        _, z = model(x_batch)
        q = clustering_layer(z)
        p = target_distribution(q).detach()
        loss = kl_loss(torch.log(q), p)
        dec_optimizer.zero_grad()
        loss.backward()
        dec_optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 29:
        with torch.no_grad():
            _, z_eval = model(X.to(device))
            kmeans_eval = KMeans(n_clusters=4).fit(z_eval.cpu().numpy())
            score = silhouette_score(z_eval.cpu().numpy(), kmeans_eval.labels_)
        print(f"[DEC] Epoch {epoch+1}, KL Loss: {total_loss:.4f}, Silhouette Score: {score:.4f}")

#Final Evaluation
model.eval()
with torch.no_grad():
    _, final_latent = model(X.to(device))
    latent_np = final_latent.cpu().numpy()
    final_clusters = KMeans(n_clusters=4).fit_predict(latent_np)

sil_score = silhouette_score(latent_np, final_clusters)
db_score = davies_bouldin_score(latent_np, final_clusters)
print(f"\n🔍 Final Silhouette Score: {sil_score:.4f}")
print(f"📉 Davies-Bouldin Index: {db_score:.4f}")

#Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_embeds = tsne.fit_transform(latent_np)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=final_clusters, cmap='tab10')
plt.colorbar(scatter, label='Cluster')
plt.title("t-SNE of Final DEC Clusters")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()


# 🔍 Final Silhouette Score: 0.9819
# 📉 Davies-Bouldin Index: 0.0257