import numpy as np
import uuid
import random
from typing import List, Tuple
from src.features.schemas import Customer, Channel, Interaction

class Simulator:
    """
    Simulates customer environment and response dynamics.
    The 'ground truth' logic is hidden here and used to generate rewards
    for the bandits to learn.
    """
    
    CHANNEL_COSTS = {
        Channel.SMS: 1.0,
        Channel.CALL: 5.0,
        Channel.FIELD_VISIT: 20.0
    }
    
    REGIONS = ["North", "South", "East", "West"]
    TIERS = ["Basic", "Standard", "Premium"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_customer(self) -> Customer:
        """Generate a random customer profile."""
        risk_score = self.rng.uniform(0, 100)
        
        # Correlate outstanding amount with risk slightly
        base_amount = self.rng.exponential(scale=5000)
        outstanding_amount = base_amount * (1 + risk_score/200)
        
        days_past_due = int(self.rng.gamma(shape=2, scale=15))
        
        # Credit score negatively correlated with risk
        credit_val = self.rng.normal(700 - risk_score*3, 50)
        credit_val = max(300, min(850, credit_val))
        # Bucket: 1 (poor) to 5 (excellent)
        if credit_val < 580: bucket = 1
        elif credit_val < 670: bucket = 2
        elif credit_val < 740: bucket = 3
        elif credit_val < 800: bucket = 4
        else: bucket = 5
        
        return Customer(
            customer_id=str(uuid.uuid4()),
            risk_score=round(risk_score, 2),
            outstanding_amount=round(outstanding_amount, 2),
            days_past_due=days_past_due,
            credit_score_bucket=bucket,
            customer_tier=self.rng.choice(self.TIERS),
            region=self.rng.choice(self.REGIONS),
            contact_history_count=int(self.rng.poisson(2)),
            last_channel=self.rng.choice([c.value for c in Channel]) if self.rng.random() > 0.3 else None
        )

    def _calculate_recovery_prob(self, customer: Customer, channel: Channel) -> float:
        """
        Hidden logic defining the 'True' probability of recovery.
        Bandit needs to learn this.
        """
        # Base probability
        prob = 0.05 
        
        # Feature impact logic (The Ground Truth)
        
        # 1. High risk customers ignore SMS, need Field
        if customer.risk_score > 80:
            if channel == Channel.SMS: prob -= 0.04
            elif channel == Channel.FIELD_VISIT: prob += 0.25
            
        # 2. Low DPD customers respond well to soft nudges (SMS)
        if customer.days_past_due < 10:
            if channel == Channel.SMS: prob += 0.15
            elif channel == Channel.FIELD_VISIT: prob -= 0.10 # Too aggressive
            
        # 3. High Outstanding Amount warrants Calls/Field
        if customer.outstanding_amount > 10000:
            if channel == Channel.SMS: prob -= 0.02
            elif channel == Channel.CALL: prob += 0.10
            
        # 4. Premium customers hate calls, prefer SMS or Field (white glove)
        if customer.customer_tier == "Premium":
            if channel == Channel.CALL: prob -= 0.05
            elif channel == Channel.SMS: prob += 0.05
            
        # Channel inherent effectiveness boost
        if channel == Channel.CALL: prob += 0.05
        if channel == Channel.FIELD_VISIT: prob += 0.10
        
        # Sigmoid-ish clipping
        return max(0.0, min(0.95, prob))

    def get_reward(self, customer: Customer, channel: Channel) -> float:
        """
        Simulate the outcome and return Net Reward.
        Net Reward = (Recovery Amount * Success Binary) - Cost
        """
        prob_success = self._calculate_recovery_prob(customer, channel)
        is_success = self.rng.random() < prob_success
        
        cost = self.CHANNEL_COSTS[channel]
        
        if is_success:
            # Recovery amount is a fraction of outstanding
            # usually full amount for small debts, partial for large
            recovery_rate = self.rng.beta(8, 2) # high recovery rate if success
            recovered = customer.outstanding_amount * recovery_rate
        else:
            recovered = 0.0
            
        return recovered - cost, is_success, recovered
