# Adapts your existing BiasClassifier/BiasCorrector into simple functions
# Place this next to server.py and point MODEL_DIR to your fine-tuned model.

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
from captum.attr import IntegratedGradients

MODEL_DIR = os.getenv("MODEL_DIR", "../fine-tuned-multiclass-bias-model")
LABEL_MAP = {
    0: "Gender Bias",
    1: "Neutral",
    2: "Political Bias",
    3: "Professional Bias",
    4: "Racial Bias",
    5: "Religious Bias"
}

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
_model.eval()

@torch.no_grad()
def predict_with_probs(text: str) -> Dict:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    out = _model(**inputs).logits.softmax(-1)[0].detach().cpu()
    pred_id = int(out.argmax().item())
    return {
        "predicted_class": LABEL_MAP.get(pred_id, "Unknown"),
        "confidence": float(out[pred_id].item()),
        "all_probabilities": {LABEL_MAP[i]: float(out[i].item()) for i in range(out.shape[0])}
    }

# Token-level attribution using Captum Integrated Gradients
from captum.attr import IntegratedGradients

_ig = IntegratedGradients(lambda input_ids, attention_mask: _model(
    input_ids=input_ids, attention_mask=attention_mask
).logits)

@torch.no_grad()
def token_attributions(text: str, target_idx: int) -> dict:
    enc = _tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)

    # Get embeddings from model
    embeddings = _model.bert.embeddings.word_embeddings(input_ids)

    # Define forward function that accepts embeddings directly
    def forward_func(embeds):
        outputs = _model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs.logits

    # Initialize Integrated Gradients on embeddings
    ig = IntegratedGradients(forward_func)

    # Compute attribution on embeddings
    attributions, delta = ig.attribute(
        embeddings,
        target=target_idx,
        n_steps=32,
        return_convergence_delta=True
    )

    # Aggregate attribution scores for each token (sum over embedding dims)
    token_attrs = attributions.sum(dim=-1).squeeze(0).abs().detach().cpu().tolist()

    # Get tokens and character offsets
    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])
    offsets = _tokenizer(text, return_offsets_mapping=True, truncation=True)["offset_mapping"]

    spans = []
    for i, (tok, score) in enumerate(zip(tokens, token_attrs)):
        if i < len(offsets):
            start, end = offsets[i]
            if end > start:
                spans.append({
                    "token": tok,
                    "start": int(start),
                    "end": int(end),
                    "score": float(score)
                })

    return {"spans": spans}