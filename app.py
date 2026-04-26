from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# -------- INIT --------
app = FastAPI()

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODEL LOAD --------
model_path = "./spam_model"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# -------- INPUT --------
class InputBatch(BaseModel):
    texts: List[str]

# -------- CLASSIFIER --------
def classify_text(text):
    text_clean = text.lower()

    inputs = tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    prob_spam = probs[0][1].item()

    patterns = []
    important_words = []

    # -------- KEYWORDS --------
    reward_words = ["free", "win", "prize", "reward", "offer"]
    urgency_words = ["urgent", "now", "immediately", "hurry", "act fast", "today", "final reminder"]
    scam_words = ["verify", "account", "bank", "blocked", "suspended"]
    money_words = ["earn", "money", "investment", "cash"]

    # -------- IMPORTANT WORDS --------
    for word in reward_words + urgency_words + scam_words + money_words:
        if word in text_clean:
            important_words.append(word)

    # -------- BANK MESSAGE --------
    bank_patterns = [
        "sent rs", "debited", "credited", "a/c", "ref",
        "transaction", "upi", "bank", "not you", "sms block", "call"
    ]
    if sum(1 for p in bank_patterns if p in text_clean) >= 3:
        return {
            "message": text,
            "prediction": "Not Spam ✅",
            "confidence": 0.96,
            "important_words": ["bank", "transaction"],
            "patterns_detected": ["Trusted bank message"],
            "explanation": "Legitimate bank transaction alert"
        }

    # -------- STOCK / TRADING --------
    trading_patterns = [
        "nse", "bse", "traded value", "broker",
        "stock", "shares", "cm rs", "registered email"
    ]
    if sum(1 for p in trading_patterns if p in text_clean) >= 2:
        return {
            "message": text,
            "prediction": "Not Spam ✅",
            "confidence": 0.96,
            "important_words": ["trading"],
            "patterns_detected": ["Trusted financial message"],
            "explanation": "Legitimate trading notification"
        }

    # -------- OTP --------
    if "otp" in text_clean and ("do not share" in text_clean or "never share" in text_clean):
        return {
            "message": text,
            "prediction": "Not Spam ✅",
            "confidence": 0.95,
            "important_words": ["otp"],
            "patterns_detected": ["Trusted system message"],
            "explanation": "Legitimate OTP message"
        }

    # -------- SERVICE / DOCUMENT --------
    service_patterns = [
        "digio", "groww", "signed a document", "esign",
        "e-sign", "document", "signed copy", "get your signed copy"
    ]
    if sum(1 for p in service_patterns if p in text_clean) >= 2:
        return {
            "message": text,
            "prediction": "Not Spam ✅",
            "confidence": 0.95,
            "important_words": ["document"],
            "patterns_detected": ["Trusted service message"],
            "explanation": "Legitimate document/service notification"
        }

    # -------- PATTERNS --------
    if any(w in text_clean for w in reward_words):
        patterns.append("Reward")

    if any(w in text_clean for w in urgency_words):
        patterns.append("Urgency")

    if any(w in text_clean for w in scam_words):
        patterns.append("Scam")

    if any(w in text_clean for w in money_words):
        patterns.append("Money")

    # -------- RULES --------

    # Reward + urgency
    if "Reward" in patterns and "Urgency" in patterns:
        return {
            "message": text,
            "prediction": "Spam 🚫",
            "confidence": 0.95,
            "important_words": important_words or ["prize"],
            "patterns_detected": patterns,
            "explanation": "Reward + urgency scam"
        }

    # Account threat
    if "Scam" in patterns and "Urgency" in patterns:
        return {
            "message": text,
            "prediction": "Spam 🚫",
            "confidence": 0.94,
            "important_words": important_words or ["account"],
            "patterns_detected": patterns,
            "explanation": "Phishing attempt detected"
        }

    # Money scam
    if "Money" in patterns:
        return {
            "message": text,
            "prediction": "Spam 🚫",
            "confidence": 0.9,
            "important_words": important_words,
            "patterns_detected": patterns,
            "explanation": "Financial scam detected"
        }

    # -------- MODEL DECISION --------
    if prob_spam > 0.6:
        prediction = "Spam 🚫"
        confidence = prob_spam
        explanation = "Model prediction"
    else:
        prediction = "Not Spam ✅"
        confidence = 1 - prob_spam
        explanation = "No strong spam indicators"

    return {
        "message": text,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "important_words": important_words or ["None"],
        "patterns_detected": patterns or ["None"],
        "explanation": explanation
    }

# -------- API --------
@app.post("/predict_batch")
def predict(data: InputBatch):
    return {"results": [classify_text(t) for t in data.texts]}

# -------- HOME --------
@app.get("/")
def home():
    return {"status": "Spam Detector running 🚀"}