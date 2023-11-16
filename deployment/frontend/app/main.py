import streamlit as st
import requests
from requests_responses import LoanFeatures
import drop_down_lists

# Define the FastAPI endpoint URL
FASTAPI_URL = "https://fee-prediction-backend-ubfu2xlf7q-oe.a.run.app/consumer_loan_fee"

st.title("Consumer Loan Fee Predictor")


st.sidebar.header("Loan Request Details")

amt_application = st.sidebar.number_input(
    "Loan Amount", min_value=0.0, value=0.0, step=100.0
)
rate_down_payment = st.sidebar.number_input(
    "Down Payment Rate", 0.0, value=0.0, step=0.1
)
name_payment_type = st.sidebar.selectbox(
    "Payment Type",
    drop_down_lists.payment_type,
)
name_client_type = st.sidebar.selectbox(
    "Client Type",
    drop_down_lists.client_type,
)
name_goods_category = st.sidebar.selectbox(
    "Goods Category",
    drop_down_lists.goods_type,
)
channel_type = st.sidebar.selectbox("Channel Type", drop_down_lists.channel)
sellerplace_area = st.sidebar.number_input(
    "Area (encoded)",
    min_value=100,
    value=10000,
)
name_seller_industry = st.sidebar.selectbox(
    "Seller Industry", drop_down_lists.seller_industry
)
cnt_payment = st.sidebar.number_input(
    "Number of Payments",
    min_value=1,
    value=1,
)
product_combination = st.sidebar.selectbox(
    "Product Combination", drop_down_lists.product_combo
)
nflag_insured_on_approval = st.sidebar.checkbox("Insured", value=True)
weekday = st.sidebar.selectbox(
    "Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
hour = st.sidebar.slider("Hour", 1, 24, 12)

# Create a button to trigger the prediction
if st.sidebar.button("Predict Fee"):
    input_data = LoanFeatures(
        AMT_ANNUITY=amt_application / cnt_payment,
        AMT_APPLICATION=amt_application,
        AMT_DOWN_PAYMENT=rate_down_payment * amt_application,
        RATE_DOWN_PAYMENT=rate_down_payment,
        NAME_PAYMENT_TYPE=name_payment_type,
        NAME_TYPE_SUITE="Other_A",
        NAME_CLIENT_TYPE=name_client_type,
        NAME_GOODS_CATEGORY=name_goods_category,
        CHANNEL_TYPE=channel_type,
        SELLERPLACE_AREA=sellerplace_area,
        NAME_SELLER_INDUSTRY=name_seller_industry,
        CNT_PAYMENT=cnt_payment,
        PRODUCT_COMBINATION=product_combination,
        NFLAG_INSURED_ON_APPROVAL=nflag_insured_on_approval,
        Weekday=weekday,
        Hour=hour,
    )

    response = requests.post(FASTAPI_URL, json=input_data.model_dump())

    if response.status_code == 200:
        suggested_fee = response.json()["SuggestedFee"]
        st.success(f"Suggested Fee: {suggested_fee}")
    else:
        st.error(f"Error making prediction. Code: {response.status_code}")
