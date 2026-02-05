from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return a @ b.T


class Embedder(nn.Module):
    def __init__(self, model, num_outputs, num_sentences):
        super(Embedder, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([0.07]))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        anchors = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        positives = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        return self.loss(anchors, positives)

    def loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        similarity_matrix = (a @ b.T) / (self.temperature + 1e-12)
        labels = torch.arange(similarity_matrix.shape[0]).to(similarity_matrix.device)
        return F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(
            similarity_matrix.T, labels
        )

    def get_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        sentence1 = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        sentence2 = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        return cosine_sim(sentence1, sentence2).diagonal()

    def embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                1
            ]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def compute_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        similarity = torch.mm(emb1, emb2.T)
        return similarity

    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: List[str] = None,
        num_samples: int = 1000,
        perplexity: int = 30,
    ) -> None:

        embeddings_np = embeddings.cpu().numpy()
        if num_samples > len(embeddings_np):
            num_samples = len(embeddings_np)
        selected_embeddings = embeddings_np[:num_samples]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(selected_embeddings)

        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c="blue", alpha=0.5)
        if labels:
            for i, label in enumerate(labels[:num_samples]):
                plt.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]))
        plt.title("t-SNE Visualization of Sentence Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()
