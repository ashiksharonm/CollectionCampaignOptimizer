from src.data.simulator import Simulator
from src.features.transformer import FeatureTransformer
from src.bandits.policies import RandomPolicy, EpsilonGreedyPolicy, LinUCBPolicy, ThompsonSamplingPolicy
from src.training.trainer import Trainer
from src.evals.evaluator import Evaluator
import numpy as np

def run_experiments():
    # Setup
    n_rounds = 2000
    simulator = Simulator(seed=42)
    transformer = FeatureTransformer()
    trainer = Trainer(simulator, transformer)
    evaluator = Evaluator()
    
    feature_dim = transformer.dim
    n_actions = 3 # SMS, Call, Field
    
    # Initialize Policies
    policies = [
        RandomPolicy(n_actions, feature_dim),
        EpsilonGreedyPolicy(n_actions, feature_dim, epsilon=0.1),
        LinUCBPolicy(n_actions, feature_dim, alpha=1.0),
        ThompsonSamplingPolicy(n_actions, feature_dim)
    ]
    
    results = []
    
    # Train
    for policy in policies:
        res = trainer.train(policy, n_rounds=n_rounds)
        results.append(res)
        
    # Evaluate
    summary = evaluator.compare_policies(results)
    evaluator.plot_action_distribution(results)
    
    print("Experiments Completed.")
    print(summary)

if __name__ == "__main__":
    run_experiments()
