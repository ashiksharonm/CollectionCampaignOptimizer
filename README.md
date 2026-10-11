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

### Observed results (synthetic simulator, 2,000 rounds)

Total-reward uplift vs `RandomPolicy`, from one run of the experiment loop with simulator seeds 42 / 1 / 7 (Apple M-series Mac, our own run):

| Policy | seed 42 | seed 1 | seed 7 |
|---|---|---|---|
| EpsilonGreedy (eps=0.1) | +49.4% | +26.2% | +19.2% |
| LinUCB (alpha=1.0) | +45.2% | +18.6% | +42.8% |
| Thompson Sampling | +47.2% | +7.2% | +58.7% |

Uplift varies widely by seed, and results are not fully reproducible: the policies' own random number generators are unseeded, so repeating seed 7 gave LinUCB +23.6% and Thompson Sampling +46.2% in an earlier run. Treat these as a rough range, not a headline number. Everything is on simulated data with a hand-written reward function, so it says nothing about real collections performance.

**Limitations of the evaluation**
- Regret is computed from noisy sampled rewards rather than expected rewards, and policies are compared on unpaired RNG streams (they do not see the same customers), so regret and uplift carry sampling noise.
- No latency or explainability-coverage benchmark is committed; earlier claims of "<10ms" and "100% transparency" were removed.

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
