import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.app import app
from src.data.simulator import Simulator
from src.features.transformer import FeatureTransformer
from src.bandits.policies import LinUCBPolicy, Channel
from src.features.schemas import Customer
import random

# Core Component Tests
def test_simulator_shape():
    sim = Simulator(seed=42)
    cust = sim.generate_customer()
    assert isinstance(cust, Customer)
    
    r, success, amount = sim.get_reward(cust, Channel.SMS)
    assert isinstance(r, float)
    assert isinstance(success, bool)

def test_manual_feature_transform():
    sim = Simulator(seed=42)
    transformer = FeatureTransformer()
    cust = sim.generate_customer()
    
    vec = transformer.transform(cust)
    assert vec.shape == (transformer.dim,)
    assert not np.isnan(vec).any()

def test_linucb_update():
    # Simple check that matrix updates
    policy = LinUCBPolicy(n_actions=2, feature_dim=5)
    ctx = np.ones(5)
    
    prev_norm = np.linalg.norm(policy.A[0])
    policy.update(ctx, Channel.SMS, 1.0)
    new_norm = np.linalg.norm(policy.A[0])
    
    # Since we added np.outer(ctx, ctx), norm should increase or change
    assert new_norm != prev_norm

# API Tests
client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_recommend():
    # We need to wait for startup if running real app, 
    # but TestClient usually runs startup events. The startup event is heavy (training).
    # This test might be slow.
    with TestClient(app) as local_client:
        customer_payload = {
            "customer_id": "test_1",
            "risk_score": 50,
            "outstanding_amount": 1000,
            "days_past_due": 10,
            "credit_score_bucket": 3,
            "customer_tier": "Standard",
            "region": "North",
            "contact_history_count": 1
        }
        
        response = local_client.post("/recommend", json=customer_payload)
        assert response.status_code == 200
        data = response.json()
        assert "recommended_channel" in data
        assert "explanation" in data
