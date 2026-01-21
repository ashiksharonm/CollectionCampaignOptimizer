from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class Channel(str, Enum):
    SMS = "SMS"
    CALL = "Call"
    FIELD_VISIT = "Field"

class Customer(BaseModel):
    customer_id: str
    risk_score: float = Field(..., ge=0, le=100, description="0-100 score, higher is riskier")
    outstanding_amount: float = Field(..., ge=0)
    days_past_due: int = Field(..., ge=0)
    credit_score_bucket: int = Field(..., ge=1, le=5, description="1-5 rating, 5 is best")
    customer_tier: str = Field(..., description="Premium, Standard, Basic")
    region: str = Field(..., description="North, South, East, West")
    contact_history_count: int = Field(0, ge=0)
    last_channel: Optional[Channel] = None

class Interaction(BaseModel):
    interaction_id: str
    customer: Customer
    selected_channel: Channel
    recovered_amount: float
    channel_cost: float
    net_reward: float
    is_successful: bool
    # For logs
    model_version: str
    prob_sms: Optional[float] = None
    prob_call: Optional[float] = None
    prob_field: Optional[float] = None
