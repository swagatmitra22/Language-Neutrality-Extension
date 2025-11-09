import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from suggestor_adapter import predict_with_probs, token_attributions, LABEL_MAP

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Language Neutrality API", version="1.0")

# ✅ FIX: Allow CORS for Chrome Extension fetch() requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow requests from all origins (Chrome ext)
    allow_credentials=True,
    allow_methods=["*"],          # allow all HTTP methods (including OPTIONS)
    allow_headers=["*"],          # allow all headers
)

# -------------------------------
# Request Models
# -------------------------------
class AnalyzeReq(BaseModel):
    text: str
    threshold: float = 0.4  # for span selection from attributions

class SuggestReq(BaseModel):
    sentence: str
    bias_type: str
    confidence: float

class CFEReq(BaseModel):
    sentence: str
    span_start: int
    span_end: int
    replacement: str

# -------------------------------
# Analyze Endpoint
# -------------------------------
@app.post("/analyze")
def analyze(req: AnalyzeReq):
    pred = predict_with_probs(req.text)

    # pick target idx for Integrated Gradients
    label_to_idx = {v: k for k, v in LABEL_MAP.items()}
    target_idx = label_to_idx.get(pred["predicted_class"], 1)

    attrs = token_attributions(req.text, target_idx)

    # select important spans (merge adjacent tokens over threshold)
    spans = []
    current = None

    # normalize scores
    scores = [s["score"] for s in attrs["spans"]]
    max_s = max(scores) if scores else 1.0

    for s in attrs["spans"]:
        norm = (s["score"] / max_s) if max_s else 0.0
        if norm >= req.threshold and s["end"] > s["start"]:
            if current and s["start"] == current["end"]:
                current["end"] = s["end"]
                current["avg_score_sum"] += norm
                current["count"] += 1
            else:
                if current:
                    current["avg_score"] = current["avg_score_sum"] / current["count"]
                    spans.append(current)
                current = {"start": s["start"], "end": s["end"], "avg_score_sum": norm, "count": 1}
        else:
            if current:
                current["avg_score"] = current["avg_score_sum"] / current["count"]
                spans.append(current)
                current = None

    if current:
        current["avg_score"] = current["avg_score_sum"] / current["count"]
        spans.append(current)

    return {
        "prediction": pred,
        "spans": spans,
    }

# -------------------------------
# Optional Gemini Suggestion Endpoint
# -------------------------------
try:
    import google.generativeai as genai
    GEN_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEN_KEY:
        genai.configure(api_key=GEN_KEY)
        gmodel = genai.GenerativeModel("gemini-2.0-flash-exp")
    else:
        gmodel = None
except Exception:
    gmodel = None

@app.post("/suggest")
def suggest(req: SuggestReq):
    if not gmodel:
        return {"suggestion": "(Gemini not configured)", "used_model": None}

    prompt = (
        f"Text: {req.sentence}\n"
        f"Bias Detected: {req.bias_type} (confidence {req.confidence:.2f})\n"
        "Rewrite this sentence to remove bias and make it neutral, "
        "without changing its meaning. Return only the rewritten version."
    )

    try:
        out = gmodel.generate_content(prompt)
        suggestion = out.text.strip() if out and out.text else "(no suggestion)"
        return {"suggestion": suggestion, "used_model": "gemini-2.0-flash-exp"}
    except Exception as e:
        print("Gemini error:", e)
        return {
            "suggestion": "(Unable to fetch suggestion — Gemini quota limit hit)",
            "used_model": "fallback",
        }

# -------------------------------
# Counterfactual Explanation Endpoint
# -------------------------------
@app.post("/counterfactual")
def counterfactual(req: CFEReq):
    # Replace the span and re-run classifier to show probability drop
    s = req.sentence
    s2 = s[:req.span_start] + req.replacement + s[req.span_end:]
    before = predict_with_probs(s)
    after = predict_with_probs(s2)
    return {"original": s, "edited": s2, "before": before, "after": after}
