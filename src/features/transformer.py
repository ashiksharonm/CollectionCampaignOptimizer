import numpy as np
import pandas as pd
from typing import List, Dict
from src.features.schemas import Customer

class FeatureTransformer:
    """
    Converts Customer objects into numerical feature vectors for Contextual Bandits.
    Simple One-Hot Encoding + Normalization.
    """
    
    def __init__(self):
        # Define mappings for categoricals
        self.tiers = ["Basic", "Standard", "Premium"]
        self.regions = ["North", "South", "East", "West"]
        
        # Calculate dimension
        # Continuous: risk_score, outstanding_amount, days_past_due, contact_history, credit_bucket (5)
        # OneHot: Tier(3), Region(4)
        # Total: 12
        self.dim = 5 + len(self.tiers) + len(self.regions)

    def transform(self, customer: Customer) -> np.ndarray:
        """
        Returns a (dim,) numpy array.
        """
        # Continuous features (simple scaling/normalization could happen here, doing raw for now)
        # Ideally, we should scale these.
        
        # Log transform outstanding amount to squish it
        log_amount = np.log1p(customer.outstanding_amount)
        
        continuous = np.array([
            customer.risk_score / 100.0,
            log_amount / 10.0, # roughly scale to 0-1 range
            customer.days_past_due / 30.0, # scale by approx month
            customer.contact_history_count / 5.0,
            customer.credit_score_bucket / 5.0
        ])
        
        # One-hot Tier
        tier_vec = np.zeros(len(self.tiers))
        if customer.customer_tier in self.tiers:
            tier_vec[self.tiers.index(customer.customer_tier)] = 1.0
            
        # One-hot Region
        region_vec = np.zeros(len(self.regions))
        if customer.region in self.regions:
            region_vec[self.regions.index(customer.region)] = 1.0
            
        return np.concatenate([continuous, tier_vec, region_vec])
    
    def get_feature_names(self) -> List[str]:
        return [
            "risk_score_norm", "log_outstanding_norm", "dpd_norm", "contacts_norm", "credit_bucket_norm"
        ] + [f"tier_{t}" for t in self.tiers] + [f"region_{r}" for r in self.regions]
