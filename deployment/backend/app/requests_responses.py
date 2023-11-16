from pydantic import BaseModel, Field
from typing import Optional


class LoanFeatures(BaseModel):
    """
    Pydantic model for consumer loan request.
    """

    AMT_ANNUITY: float
    AMT_APPLICATION: float
    AMT_DOWN_PAYMENT: float
    RATE_DOWN_PAYMENT: float
    NAME_PAYMENT_TYPE: str
    NAME_TYPE_SUITE: str
    NAME_CLIENT_TYPE: Optional[str] = Field(..., nullable=True)
    NAME_GOODS_CATEGORY: str
    CHANNEL_TYPE: str
    SELLERPLACE_AREA: float
    NAME_SELLER_INDUSTRY: str
    CNT_PAYMENT: int
    PRODUCT_COMBINATION: str
    NFLAG_INSURED_ON_APPROVAL: int
    Weekday: str
    Hour: int


class FeeResponse(BaseModel):
    """
    Pydantic model for fee suggestion responses.
    """

    SuggestedFee: str
