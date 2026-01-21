import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from src.features.schemas import Customer, Interaction, Channel
from src.features.transformer import FeatureTransformer
from src.data.simulator import Simulator
from src.bandits.policies import BasePolicy

class Trainer:
    """
    Manages the training loop:
    1. Generator Customer
    2. Policy selects Action
    3. Simulator returns Reward
    4. Policy updates
    """
    def __init__(self, simulator: Simulator, transformer: FeatureTransformer):
        self.simulator = simulator
        self.transformer = transformer
        
    def train(self, policy: BasePolicy, n_rounds: int = 1000) -> Dict[str, Any]:
        rewards = []
        regrets = []
        actions = []
        
        # Calculate optimal reward for regret
        # This is expensive, so maybe we cheat and ask the simulator for best action
        
        for t in tqdm(range(n_rounds), desc=f"Training {policy.get_model_name()}"):
            customer = self.simulator.generate_customer()
            context = self.transformer.transform(customer)
            
            # Policy Action
            action = policy.select_action(context)
            actions.append(action)
            
            # Real Reward
            reward, is_success, recovered = self.simulator.get_reward(customer, action)
            rewards.append(reward)
            
            # Oracle (Best Possible Reward) for Regret
            # Check all channels
            best_reward = -float('inf')
            for ch in Channel:
                r, _, _ = self.simulator.get_reward(customer, ch)
                # Note: This is stochastic, so 'best_reward' is noisy. 
                # Ideally we want Expected Reward. 
                # Simulator's _calculate_recovery_prob is hidden, effectively we are doing approximate regret.
                if r > best_reward:
                    best_reward = r
            
            regrets.append(best_reward - reward)
            
            # Update Policy
            policy.update(context, action, reward)
            
        return {
            "model": policy.get_model_name(),
            "rewards": rewards,
            "regrets": regrets,
            "actions": actions,
            "cumulative_reward": np.cumsum(rewards),
            "cumulative_regret": np.cumsum(regrets)
        }
