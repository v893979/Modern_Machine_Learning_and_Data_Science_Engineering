"""
Chapter 13: MLOps: Serving Production ML Models via FastAPI & Pydantic
Modern Machine Learning & Data Science Engineering
"""
import sys

# Simulation script illustrating how a production REST API is structured
FASTAPI_CODE = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(
    title="Modern ML Credit Scoring Service",
    version="1.0.0",
    description="Production-grade REST API for real-time credit default risk inference"
)

# 1. Pydantic Request Contract
class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Annual income in USD")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_int_rate: float = Field(..., ge=1.0, le=40.0, description="Interest rate percentage")
    cb_person_default_on_file: str = Field(..., pattern="^(Y|N)$", description="'Y' or 'N'")

# 2. Pydantic Response Contract
class PredictionResponse(BaseModel):
    default_probability: float
    credit_risk_tier: str
    underwriting_decision: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "credit-risk-engine", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(app_data: LoanApplication):
    # Calculate automated risk score
    dti = app_data.loan_amnt / app_data.person_income
    prob = float(np.clip((dti * 0.45) + (app_data.loan_int_rate / 35.0), 0.02, 0.95))
    
    tier = "HIGH RISK" if prob > 0.40 else "MEDIUM RISK" if prob > 0.20 else "LOW RISK"
    decision = "DECLINE" if prob > 0.35 else "APPROVE"
    
    return {
        "default_probability": round(prob, 4),
        "credit_risk_tier": tier,
        "underwriting_decision": decision
    }
"""

def main():
    print("=====================================================================")
    print("   PRODUCTION MLOPS: FASTAPI REST MICROSERVICE ARCHITECTURE          ")
    print("=====================================================================")
    print("This script provides the complete template for serving models via FastAPI.")
    print("\nFastAPI Application Code Snippet:")
    print(FASTAPI_CODE)
    print("To launch server live: uvicorn ch13_fastapi_model_deployment:app --reload --port 8000")

if __name__ == "__main__":
    main()
