

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin

# -------------------------------------------------------------------- config
INPUT_PATH  = Path("state_output_sample1000.jsonl")
OUTPUT_JSON = Path("domain_results.json")
OUTPUT_TXT  = Path("domain_summary.txt")

MODEL_NAME  = "nvidia/multilingual-domain-classifier"
N_DOCS      = 20       # number of documents to classify
BATCH_SIZE  = 16
MAX_CHARS   = 2000     # truncate text to first 2000 chars (model recommendation)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ model def
class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.model   = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc      = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped  = self.dropout(features)
        outputs  = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


# ------------------------------------------------------------------ IO helpers
def load_records(path: Path, n: int) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) == n:
                break
    return records


# ------------------------------------------------------------------ inference
def classify_domains(records: list[dict]) -> list[dict]:
    print(f"[domain] loading model: {MODEL_NAME}")
    config    = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = CustomModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE).float()

    id2label = config.id2label
    print(f"[domain] model loaded on {DEVICE}  |  {len(id2label)} classes")
    print(f"[domain] classifying {len(records)} documents...")

    results = []
    t0 = time.time()

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        texts = [
            (r.get("record", {}).get("text") or "")[:MAX_CHARS]
            for r in batch
        ]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            probs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        for i, r in enumerate(batch):
            pred_id    = probs[i].argmax().item()
            pred_label = id2label[pred_id]
            confidence = probs[i][pred_id].item()
            results.append({
                "url":        r.get("record", {}).get("url"),
                "domain":     pred_label,
                "confidence": round(confidence, 4),
                "top3": [
                    {"label": id2label[j], "score": round(probs[i][j].item(), 4)}
                    for j in probs[i].topk(3).indices.tolist()
                ],
            })

    print(f"[domain] done in {time.time() - t0:.1f}s")
    return results


# ------------------------------------------------------------------ summary
def print_and_save_summary(results: list[dict], out_path: Path) -> None:
    counter  = collections.Counter(r["domain"] for r in results)
    total    = len(results)
    avg_conf = sum(r["confidence"] for r in results) / total

    lines = []
    lines.append("=" * 60)
    lines.append("DOMAIN DISTRIBUTION — nvidia/multilingual-domain-classifier")
    lines.append(f"Documents classified : {total}")
    lines.append(f"Average confidence   : {avg_conf:.3f}")
    lines.append("=" * 60)
    for domain, count in counter.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {domain:<30} {count:>3}  ({pct:5.1f}%)  {bar}")
    lines.append("=" * 60)
    lines.append("\nPer-document predictions:")
    lines.append(f"  {'URL':<55} {'Domain':<30} Conf")
    lines.append("  " + "-" * 95)
    for r in results:
        url = (r["url"] or "")[:55]
        lines.append(f"  {url:<55} {r['domain']:<30} {r['confidence']:.3f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    out_path.write_text(summary, encoding="utf-8")
    print(f"\n[domain] saved summary → {out_path}")


# ---------------------------------------------------------------------- main
def main() -> None:
    print(f"[domain] reading {N_DOCS} documents from: {INPUT_PATH}")
    records = load_records(INPUT_PATH, N_DOCS)
    print(f"[domain] loaded {len(records)} records")

    results = classify_domains(records)

    OUTPUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[domain] saved predictions → {OUTPUT_JSON}")

    print_and_save_summary(results, OUTPUT_TXT)


if __name__ == "__main__":
    main()