import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="SympTriage API")

# Input model: user sends a list of messages
class Input(BaseModel):
    user_inputs: list[str]

# --------- Symptom extraction ---------
def extract_symptoms(state):
    if not state["user_input"].strip():
        return state

    prompt = f"""
    Extract medical symptoms from this input: {state['user_input']}.
    If severity (like 'severe', 'mild') is mentioned, include as 'symptom: severity'.
    If measurable value (like fever) is mentioned, include as 'symptom: value' (e.g., fever: 102°F).
    Return a comma-separated list only, no extra text.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    text = response.choices[0].message.content
    new_symptoms = [s.strip() for s in text.split(",") if s.strip()]

    for s in new_symptoms:
        if s.lower() not in [sym.lower() for sym in state["symptoms"]]:
            state["symptoms"].append(s)

    if state["user_input"].strip():
        state["conversation_history"].append(f"User: {state['user_input']}")

    return state

# --------- Fever check (Fahrenheit) ---------
def check_fever_degree(state):
    has_fever = any(s.lower().startswith("fever") for s in state["symptoms"])
    has_value = any("fever:" in s.lower() for s in state["symptoms"])

    # Ask for fever value only if not already provided
    if has_fever and not has_value and state.get("fever_input"):
        val = state["fever_input"]
        state["user_input"] = f"fever: {val}°F"
        state = extract_symptoms(state)

    return state

# --------- Ambiguous symptom clarifier ---------
def clarify_ambiguous_symptoms(state):
    ambiguous_terms = ["pain", "fever", "cough", "headache", "nausea", "dizziness", "vomit", "shivering"]

    if "clarified_symptoms" not in state:
        state["clarified_symptoms"] = set()

    for s in list(state["symptoms"]):
        base_symptom = s.split(":")[0].strip().lower()

        if base_symptom in ambiguous_terms and base_symptom not in state["clarified_symptoms"] and ":" not in s:
            # Skip fever if numeric value already exists
            if base_symptom == "fever" and any("fever:" in sym.lower() for sym in state["symptoms"]):
                state["clarified_symptoms"].add("fever")
                continue

            # Use user_provided clarification if available
            if state.get("clarifications") and base_symptom in state["clarifications"]:
                answer = state["clarifications"][base_symptom]
                state["user_input"] = f"{base_symptom}: {answer}"
                state = extract_symptoms(state)

            state["clarified_symptoms"].add(base_symptom)

    return state

# --------- Disease prediction ---------
def predict_disease(state):
    prompt = f"""
    The user has reported these symptoms: {', '.join(state['symptoms'])}.
    Provide:
    Disease: <name>
    Confidence: <number>%
    Triage: <level> (Emergency / Urgent / Mild)
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    output_text = response.choices[0].message.content.strip()

    disease, confidence, triage = None, None, None
    for line in output_text.split("\n"):
        if line.startswith("Disease:"):
            disease = line.split("Disease:")[1].strip()
        elif line.startswith("Confidence:"):
            confidence = float(line.split("Confidence:")[1].strip().replace("%",""))
        elif line.startswith("Triage:"):
            triage = line.split("Triage:")[1].strip()

    state.update({
        "disease_prediction": disease,
        "confidence": confidence,
        "triage_level": triage
    })

    return state

# --------- FastAPI endpoint ---------
@app.post("/predict")
def predict(input: Input):
    state = {
        "user_input": "",
        "symptoms": [],
        "conversation_history": [],
        "clarified_symptoms": set(),
        "clarifications": {},  # Optional dict: {"fever": "101°F", "cough": "severe"}
        "fever_input": None,   # Optional: numeric fever value
    }

    # Iterate over user messages
    for user_text in input.user_inputs:
        state["user_input"] = user_text
        state = extract_symptoms(state)
        state = check_fever_degree(state)
        state = clarify_ambiguous_symptoms(state)

    # Predict disease
    state = predict_disease(state)

    return {
        "conversation_history": state["conversation_history"],
        "symptoms": state["symptoms"],
        "disease_prediction": state["disease_prediction"],
        "confidence": state["confidence"],
        "triage_level": state["triage_level"]
    }

# --------- Health check ---------
@app.get("/")
def home():
    return {"message": "SympTriage API is running"}
