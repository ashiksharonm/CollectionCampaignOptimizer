# Collection Campaign Optimizer

**End-to-End Contextual Multi-Armed Bandit System for Debt Collection Optimization**

This project implements a Reinforcement Learning approach to maximize debt recovery while minimizing costs. It selects the optimal communication channel (SMS, Call, Field Visit) for each customer based on their context (risk score, days past due, etc.) using **LinUCB** and **Thompson Sampling**.

## 🏗 Architecture

```ascii
+----------------+        +------------------+         +-----------------+
| Customer Data  | -----> | Feature          | ----->  | Bandit Policy   |
| (Context)      |        | Transformer      |         | (LinUCB/TS)     |
+----------------+        +------------------+         +-----------------+
                                                              |
                                                              v
+----------------+        +------------------+         +-----------------+
|   Explainability| <-----| Recommendation   | <------ | Action Selection|
|   (SHAP)       |        | (Channel)        |         | (SMS/Call/Field)|
+----------------+        +------------------+         +-----------------+
                                                              |
                                                              v
                                                       +-----------------+
                                                       | Simulator / Environment |
                                                       | Reward = Recovery - Cost|
                                                       +-----------------+
```

## 🚀 Features

- **Synthetic Simulator**: Realistic data generator with hidden ground truth logic for recovery probabilities.
- **Bandit Algorithms**:
  - `RandomPolicy` (Baseline)
  - `EpsilonGreedy`
  - `LinUCB` (Disjoint)
  - `ThompsonSampling` (Gaussian)
- **Production API**: FastAPI service that trains on startup (warm-start) and serves recommendations.
- **Explainability**: SHAP-based explanations for why a specific channel was recommended.
- **Metrics**: Tracks Cumulative Reward, Regret, and Uplift.

## 🛠 Setup

**Prerequisites**: Python 3.11+

1. **Install Dependencies**
   ```bash
   make install
   ```

2. **Run Tests**
   ```bash
   make test
   ```

## 🔬 Training & Simulation

Run the experiment script to train bandits against the simulator and generate an evaluation report.

```bash
python run_experiments.py
```

Results will be saved to `reports/`.

## ⚡ API Usage

The API trains a model on startup using the simulator (Warm Start) so it's ready to serve immediately.

1. **Start the API**
   ```bash
   make run-api
   ```

2. **Get Recommendation**
   ```bash
   curl -X POST "http://localhost:8000/recommend" \
        -H "Content-Type: application/json" \
        -d '{
              "customer_id": "cust_123",
              "risk_score": 85.5,
              "outstanding_amount": 5000.0,
              "days_past_due": 45,
              "credit_score_bucket": 2,
              "customer_tier": "Standard",
              "region": "West",
              "contact_history_count": 3
            }'
   ```

   **Response**:
   ```json
   {
     "customer_id": "cust_123",
     "recommended_channel": "Field",
     "model_version": "1.0.0",
     "explanation": {
       "predicted_reward": 450.2,
       "top_features": [
         {"feature": "risk_score_norm", "impact": 0.35},
         {"feature": "dpd_norm", "impact": 0.12}
       ]
     }
   }
   ```

## 🐳 Docker

```bash
docker-compose up --build
```

Access API at `http://localhost:8000`.

## 📈 Future Improvements
- Integrate real feedback loop for Online Learning (POST /feedback endpoint).
- Implement categorical embeddings instead of One-Hot.
- Add A/B testing framework support.
