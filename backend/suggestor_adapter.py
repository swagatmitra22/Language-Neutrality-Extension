# Adapts your existing BiasClassifier/BiasCorrector into simple functions
# Uses MODEL_DIR from the root .env file only.

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
from captum.attr import IntegratedGradients
from dotenv import load_dotenv

# --------------------------------------------------------
# ✅ Always load environment variables from root .env only
# --------------------------------------------------------
ROOT_ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(ROOT_ENV_PATH)

# --------------------------------------------------------
# Model directory and label mapping
# --------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./fine-tuned-multiclass-bias-model")

LABEL_MAP = {
    0: "Gender Bias",
    1: "Neutral",
    2: "Political Bias",
    3: "Professional Bias",
    4: "Racial Bias",
    5: "Religious Bias"
}

# --------------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
    _model.eval()
    print(f"✅ Model loaded successfully from {MODEL_DIR} on {_device}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model from {MODEL_DIR}: {e}")

# --------------------------------------------------------
# Prediction function
# --------------------------------------------------------
@torch.no_grad()
def predict_with_probs(text: str) -> Dict:
    """
    Predicts the bias class for a given text and returns confidence scores.
    """
    if not text.strip():
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "all_probabilities": {}
        }

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    logits = _model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].detach().cpu()
    pred_id = int(probs.argmax().item())

    return {
        "predicted_class": LABEL_MAP.get(pred_id, "Unknown"),
        "confidence": float(probs[pred_id].item()),
        "all_probabilities": {LABEL_MAP[i]: float(probs[i].item()) for i in range(len(LABEL_MAP))}
    }

# --------------------------------------------------------
# Token-level attributions using Captum Integrated Gradients
# --------------------------------------------------------
@torch.no_grad()
def token_attributions(text: str, target_idx: int) -> dict:
    """
    Computes token-level attributions using Integrated Gradients
    to highlight bias-inducing words or phrases.
    """
    if not text.strip():
        return {"spans": []}

    enc = _tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)

    # Get word embeddings from model
    embeddings = _model.bert.embeddings.word_embeddings(input_ids)

    # Define forward function accepting embeddings directly
    def forward_func(embeds):
        outputs = _model(inputs_embeds=embeds, attention_mask=attention_mask)
        return outputs.logits

    ig = IntegratedGradients(forward_func)

    # Compute attributions
    attributions, _ = ig.attribute(
        embeddings,
        target=target_idx,
        n_steps=32,
        return_convergence_delta=True
    )

    # Aggregate attribution scores across embedding dimensions
    token_scores = attributions.sum(dim=-1).squeeze(0).abs().detach().cpu().tolist()

    # Map tokens to character offsets
    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])
    offsets = _tokenizer(text, return_offsets_mapping=True, truncation=True)["offset_mapping"]

    spans = []
    for i, (token, score) in enumerate(zip(tokens, token_scores)):
        if i < len(offsets):
            start, end = offsets[i]
            if end > start:
                spans.append({
                    "token": token,
                    "start": int(start),
                    "end": int(end),
                    "score": float(score)
                })

    return {"spans": spans}
