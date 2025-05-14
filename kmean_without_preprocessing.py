import pandas as pd

splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet'
}

df = pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["train"])
texts = df['text'].tolist()


from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)


import torch
import torch.nn as nn

class TextAutoencoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=64):
        super(TextAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z
    


from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor(embeddings, dtype=torch.float32)
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = TextAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        reconstructed, _ = model(x_batch)
        loss = criterion(reconstructed, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



model.eval()
with torch.no_grad():
    _, latent_vectors = model(X.to(device))
    latent_vectors = latent_vectors.cpu().numpy()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=4, random_state=42)
final_clusters = kmeans.fit_predict(latent_np)

score = silhouette_score(latent_np, final_clusters)
db_score = davies_bouldin_score(latent_np, final_clusters)
print("Silhouette Score:", score)
print(f"📉 Davies-Bouldin Index: {db_score:.4f}")

# Visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeds = tsne.fit_transform(latent_np)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_embeds[:, 0], tsne_embeds[:, 1], c=final_clusters, cmap='tab10')
plt.colorbar(scatter, label='Cluster')
plt.title("t-SNE of Final DEC Clusters")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()


# Silhouette Score: 0.059319388
# 📉 Davies-Bouldin Index: 3.6059