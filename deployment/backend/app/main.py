from model_files.model import model_fees
from fastapi import FastAPI
from auxiliary.transformers import (
    weekday_cyclic_features,
    hour_cyclic_features,
)
import polars as pl
from requests_responses import LoanFeatures, FeeResponse

app = FastAPI()


@app.post("/consumer_loan_fee", response_model=FeeResponse)
async def predict_endpoint(data: LoanFeatures):
    """
    Predict endpoint that makes predictions based on input data.

    Args:
        data (LoanDataInitial): The input data for prediction.

    Returns:
        PredictionResponse: The prediction results.
    """
    data_df = pl.DataFrame(data.model_dump())
    data_df = weekday_cyclic_features(data_df, "Weekday")
    data_df = hour_cyclic_features(data_df, "Hour")

    prediction = round(model_fees.predict(data_df)[0], 1)
    output = f"{data_df['AMT_APPLICATION'][0]*prediction/100} ({prediction} %)"
    return {"SuggestedFee": output}
