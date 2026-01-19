import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm.auto import tqdm
import json

import preprocessing
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
logging.set_verbosity_error()

print(">>> SCRIPT LOADED <<<")


try:
    with open("topic_keywords_final.json", "r", encoding="utf-8") as f:
        TOPIC_KEYWORDS = json.load(f)
    print("Loaded topic keywords from topic_keywords_final.json")
except FileNotFoundError:
    print("topic_keywords_final.json not found, using default TOPIC_KEYWORDS")
    TOPIC_KEYWORDS = {
        "code": [
            "code", "python", "java", "c++", "debugging",
            "programming", "script", "error", "html", "css"
        ],
        "math": [
            "math", "calculus", "algebra", "equation", "solve",
            "formula", "geometry", "derivative", "integral"
        ],
        "writing": [
            "essay", "email", "draft", "revise", "grammar",
            "edit", "cover letter", "resume", "outline", "summary"
        ],
        "research": [
            "citation", "reference", "source", "paper", "article",
            "search", "study", "academic", "verify"
        ],
        "chat": [
            "conversation", "chat", "joke", "fun", "advice",
            "roleplay", "friend", "casual"
        ],
    }


class HybridDataset(Dataset):
    def __init__(self, texts, tabular_data, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tabular_feats": self.tabular_data[idx],
            "labels": self.labels[idx]
        }



class HybridDistilBERT(nn.Module):
    def __init__(self, encoder, tabular_input_dim, num_labels, dropout=0.4):
        super().__init__()
        self.encoder = encoder
        self.text_hidden_size = encoder.config.hidden_size  # 768
        
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_dim = self.text_hidden_size + 128
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, tabular_feats, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]
        tab_emb = self.tabular_mlp(tabular_feats)

        combined = torch.cat((text_emb, tab_emb), dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss



def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            feats     = batch["tabular_feats"].to(device)
            labels    = batch["labels"].to(device)

            logits, _ = model(input_ids, mask, feats, labels=None)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, f1, cm

def main():
    print("Loading and preprocessing data...")
    df = pd.read_csv("training_data_clean.csv")

    X_tabular, y_all, config = preprocessing.build_features(df, TOPIC_KEYWORDS)

    df["combined_text"] = df.apply(preprocessing.combine_text_row, axis=1)
    raw_texts = df["combined_text"].to_numpy()

    student_ids = df[preprocessing.STUDENT_ID_COL].to_numpy()
    indices = np.arange(len(df))

    train_idx, _, val_idx, _, test_idx, _ = preprocessing.grouped_train_val_test_split(
        indices, indices, student_ids, val_size=0.2, test_size=0.2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

    for param in base_encoder.parameters():
        param.requires_grad = False
    for layer in base_encoder.transformer.layer[-4:]:
        for p in layer.parameters():
            p.requires_grad = True

    tabular_dim = X_tabular.shape[1]
    num_labels = len(np.unique(y_all))

    model = HybridDistilBERT(base_encoder, tabular_dim, num_labels)
    model.to(device)

    train_ds = HybridDataset(raw_texts[train_idx], X_tabular[train_idx], y_all[train_idx], tokenizer)
    val_ds   = HybridDataset(raw_texts[val_idx],   X_tabular[val_idx],   y_all[val_idx],   tokenizer)
    test_ds  = HybridDataset(raw_texts[test_idx],  X_tabular[test_idx],  y_all[test_idx],  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder") and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters() if not n.startswith("encoder") and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,   "lr": 1e-3},
            {"params": encoder_params, "lr": 1e-5}
        ]
    )


    best_val_acc = 0.0
    best_state = None
    patience = 4
    patience_counter = 0

    for epoch in range(1, 21):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            feats     = batch["tabular_feats"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, mask, feats, labels)
            loss.backward()
            optimizer.step()

        val_acc, _, _, _, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch} | Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "best_hybrid_model.pt")
            print("  >>> Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished.")


    model.load_state_dict(best_state)
    model.to(device)

    # ---- Validation ----
    val_acc, val_prec, val_rec, val_f1, val_cm = evaluate_model(model, val_loader, device)
    print("\n=== Validation Metrics ===")
    print("Val Acc :", val_acc)
    print("Macro Precision:", val_prec)
    print("Macro Recall   :", val_rec)
    print("Macro F1       :", val_f1)

    # ---- Test ----
    test_acc, test_prec, test_rec, test_f1, test_cm = evaluate_model(model, test_loader, device)
    print("\n=== TEST Metrics (Final Model) ===")
    print("Test Acc :", test_acc)
    print("Macro Precision:", test_prec)
    print("Macro Recall   :", test_rec)
    print("Macro F1       :", test_f1)
    print("\n=== Confusion Matrix (Test) ===")
    print(test_cm)
    print("\n===== Copy to Table 1 (Validation) =====")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Precision (Macro): {val_prec:.4f}")
    print(f"Recall (Macro): {val_rec:.4f}")
    print(f"Macro-F1: {val_f1:.4f}")
    print("\n===== Copy to Table 3 (TEST) =====")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision (Macro): {test_prec:.4f}")
    print(f"Recall (Macro): {test_rec:.4f}")
    print(f"Macro-F1: {test_f1:.4f}")
    print("\n===== Copy to Table 2 (Confusion Matrix) =====")
    print(test_cm)


if __name__ == "__main__":
    main()
