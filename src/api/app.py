from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uvicorn
import joblib
import os

from src.features.schemas import Customer, Channel
from src.features.transformer import FeatureTransformer
from src.bandits.policies import LinUCBPolicy
from src.data.simulator import Simulator
from src.training.trainer import Trainer
from src.explainability.explainer import Explainer
from src.utils.logger import get_logger

# Config
MODEL_VERSION = "1.0.0"
WARMUP_ROUNDS = 500

logger = get_logger("api")

app = FastAPI(title="Collection Campaign Optimizer API", version=MODEL_VERSION)

# Global State
model_state = {
    "policy": None,
    "transformer": None,
    "explainer": None,
    "ready": False
}

class RecommendationResponse(BaseModel):
    customer_id: str
    recommended_channel: Channel
    model_version: str
    explanation: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up... Warming up model.")
    
    # Initialize components
    simulator = Simulator(seed=42)
    transformer = FeatureTransformer()
    
    # We use LinUCB as our champion model
    policy = LinUCBPolicy(n_actions=3, feature_dim=transformer.dim, alpha=1.0)
    
    # Warm start training
    trainer = Trainer(simulator, transformer)
    results = trainer.train(policy, n_rounds=WARMUP_ROUNDS)
    
    # Train Explainer
    explainer = Explainer(transformer)
    contexts = np.array([transformer.transform(simulator.generate_customer()) for _ in range(WARMUP_ROUNDS)])
    # We need actions and rewards aligned. 
    # For simplicity in this demo, we re-use the training data collected during warm-up.
    # Trainer returns lists, we need to reconstruct context list from it? 
    # Actually trainer.train returns 'actions' and 'rewards'. 
    # We need to regenerate contexts or store them in trainer. 
    # Let's just generate fresh data for explainer to be safe and simple.
    
    logger.info("Training proxy explainer...")
    X_explain = []
    actions_explain = []
    rewards_explain = []
    
    for _ in range(200):
        cust = simulator.generate_customer()
        ctx = transformer.transform(cust)
        action = policy.select_action(ctx)
        reward, _, _ = simulator.get_reward(cust, action)
        
        X_explain.append(ctx)
        actions_explain.append(action)
        rewards_explain.append(reward)
        
    explainer.fit(np.array(X_explain), actions_explain, rewards_explain)
    
    # Update global state
    model_state["policy"] = policy
    model_state["transformer"] = transformer
    model_state["explainer"] = explainer
    model_state["ready"] = True
    
    logger.info("Model warmup complete. Service ready.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_ready": model_state["ready"], "version": MODEL_VERSION}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(customer: Customer, include_explanation: bool = True):
    if not model_state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    start_time = time.time()
    
    transformer = model_state["transformer"]
    policy = model_state["policy"]
    explainer = model_state["explainer"]
    
    # 1. Transform
    context = transformer.transform(customer)
    
    # 2. Predict
    action = policy.select_action(context)
    
    # 3. Explain
    explanation = None
    if include_explanation:
        explanation = explainer.explain_prediction(context, action)
        
    # Log the request
    log_data = {
        "event_type": "recommendation",
        "customer_id": customer.customer_id,
        "recommended_channel": action,
        "latency_ms": round((time.time() - start_time) * 1000, 2)
    }
    # Hack to pass extra args to logger
    extra = {"props": log_data}
    logger.info("Recommendation served", extra=extra)
    
    return RecommendationResponse(
        customer_id=customer.customer_id,
        recommended_channel=action,
        model_version=MODEL_VERSION,
        explanation=explanation
    )

@app.post("/batch_recommend", response_model=List[RecommendationResponse])
def batch_recommend(customers: List[Customer]):
    return [get_recommendation(c) for c in customers]

# Need to import numpy for startup
import numpy as np
