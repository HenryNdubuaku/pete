import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score


class GLUEWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        num_outputs: int = 0,
        num_sentences: int = 2,
    ):
        super(GLUEWrapper, self).__init__()
        self.model = model
        self.num_outputs = num_outputs
        self.num_sentences = num_sentences

        if self.num_outputs > 0:
            self.classifier = nn.Linear(self.model.d_model * num_sentences, num_outputs)
            self.temperature = nn.Parameter(torch.tensor([0.07]))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        if self.num_sentences == 1:
            return self.forward_one_sentence(batch)

        anchors = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        positives = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        labels = batch[-1].long()

        if self.num_outputs == 0:
            return self.correlation_loss(anchors, positives, labels)

        return self.classification_loss(anchors, positives, labels)

    def forward_one_sentence(self, batch: torch.Tensor) -> torch.Tensor:
        embedding = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        labels = batch[-1].long()
        return self.classification_loss_one_sentence(embedding, labels)

    def get_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        if self.num_outputs == 0:
            return self.correlation_predictions(batch)
        return self.classification_predictions(batch)

    def classification_loss(
        self,
        sentence1: torch.Tensor,
        sentence2: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:

        combined_rep = torch.cat([sentence1, sentence2], dim=-1)
        logits = self.classifier(combined_rep)
        return F.cross_entropy(logits, ground_truth)

    def classification_loss_one_sentence(
        self, sentence: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        logits = self.classifier(sentence)
        return F.cross_entropy(logits, ground_truth)

    def classification_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        if self.num_sentences == 1:
            outputs = self.model(input_ids=batch[0], attention_mask=batch[1])
            sentence = outputs[1]
            return F.softmax(self.classifier(sentence), dim=-1).argmax(dim=-1)

        outputs1 = self.model(input_ids=batch[0], attention_mask=batch[1])
        outputs2 = self.model(input_ids=batch[2], attention_mask=batch[3])
        sentence1 = outputs1[1]
        sentence2 = outputs2[1]
        combined_rep = torch.cat([sentence1, sentence2], dim=-1)
        logits = self.classifier(combined_rep)
        return F.softmax(logits, dim=-1).argmax(dim=-1)

    def correlation_loss(
        self,
        sentence1: torch.Tensor,
        sentence2: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:

        similarities = cosine_sim(sentence1, sentence2).diagonal()
        predicted_correlation = pearson_r(similarities, ground_truth)
        return 1 - predicted_correlation

    def correlation_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        sentence1 = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        sentence2 = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        return cosine_sim(sentence1, sentence2).diagonal()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot_product = torch.einsum("ij,kj->ik", a, b)
    norm_a = torch.sqrt(torch.einsum("ij,ij->i", a, a))
    norm_b = torch.sqrt(torch.einsum("ij,ij->i", b, b))
    norm_product = torch.einsum("i,j->ij", norm_a, norm_b)
    return dot_product / norm_product


def pearson_r(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered**2)) * torch.sqrt(
        torch.sum(y_centered**2)
    )
    return numerator / (denominator + eps)


GLUE_LABEL_MAPS = {
    "cola": {0: "unacceptable", 1: "acceptable"},
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "mnli": {0: "contradiction", 1: "entailment", 2: "neutral"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "wnli": {0: "not_entailment", 1: "entailment"},
    # STS-B: regression, so there's no label map (use None).
    "stsb": None,
}


def evaluate(
    model: nn.Module,
    data_loader: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    dataset_name: str,
    run_name: str,
    test: bool = False,
    glue_submission: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Evaluate or run inference on a model for a given dataset (train/validation/test).

    If 'test' is True, loads the model from weights/<run_name>.pt, obtains predictions on
    the test set, and saves results to:
      - results/<dataset_name>_<run_name>.npy        (raw predictions)
      - results/<dataset_name>_<run_name>.tsv        (GLUE-submission format, if glue_submission=True)

    Otherwise, evaluates on the 'validation' set and returns metrics (accuracy, precision, f1, mcc).
    For STS-B, returns the Spearman correlation using the spearman_evaluate function above.
    """
    model.to(device)
    model.eval()

    if test:
        # ====================
        # Inference on Test
        # ====================
        all_predictions = []

        # Load trained weights
        weight_path = f"weights/{run_name}.pt"
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in data_loader[dataset_name]["test"]:
                    batch = [x.to(device) for x in batch]
                    preds = model.get_predictions(batch).float().cpu().numpy()
                    all_predictions.append(preds)

        all_predictions = np.concatenate(all_predictions, axis=0)

        if glue_submission:
            label_map = GLUE_LABEL_MAPS.get(dataset_name, None)

            if label_map is None:
                # STS-B or unsupported -> treat as regression (float) or direct output
                # For STS-B, each row is just a float. Typical GLUE submission:
                #   index [TAB] prediction
                df = pd.DataFrame(
                    {
                        "index": range(len(all_predictions)),
                        "prediction": all_predictions.reshape(-1),
                    }
                )
            else:
                # Classification -> map integer predictions to string labels
                # If model outputs shape = (N,) or (N,1), ensure it’s flattened
                all_predictions = all_predictions.reshape(-1)

                # Map each integer prediction to a string label
                # e.g., [0,1,2] -> ["contradiction","entailment","neutral"]
                num_labels = len(label_map)
                clamped_preds = [max(0, min(int(p), num_labels - 1)) for p in all_predictions]
                str_labels = [label_map[p_int] for p_int in clamped_preds]
                df = pd.DataFrame(
                    {"index": range(len(str_labels)), "prediction": str_labels}
                )

            tsv_path = f"results/{dataset_name}_{run_name}.tsv"
            df.to_csv(tsv_path, sep="\t", index=False)
            print(f"GLUE-style TSV submission saved to {tsv_path}")

        return

    # =====================
    # Evaluation on Val
    # =====================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in data_loader[dataset_name]["validation"]:
                batch = [x.to(device) for x in batch]
                preds = model.get_predictions(batch)
                labels = batch[-1].long()

                all_preds.append(preds.float().cpu())
                all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).float().numpy()
    all_labels = torch.cat(all_labels).float().numpy()

    if dataset_name == "stsb":
        return spearman_evaluate(all_preds, all_labels)

    num_classes = len(np.unique(all_labels))
    avg_type = "macro"

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    metrics = {"accuracy": accuracy, "precision": precision, "f1": f1, "mcc": mcc}
    return metrics


def spearman_evaluate(
    similarities: np.array,
    labels: np.array,
) -> Optional[Dict[str, float]]:

    eval_pearson_cosine, _ = pearsonr(similarities, labels)
    eval_spearman_cosine, _ = spearmanr(similarities, labels)

    return {"pearsonr": eval_pearson_cosine, "spearmanr": eval_spearman_cosine}
