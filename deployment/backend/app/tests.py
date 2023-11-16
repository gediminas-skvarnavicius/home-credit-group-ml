from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)

test_data = {
    "AMT_ANNUITY": 3993.57,
    "AMT_APPLICATION": 15075.0,
    "AMT_DOWN_PAYMENT": 1507.5,
    "RATE_DOWN_PAYMENT": 0.105752,
    "NAME_PAYMENT_TYPE": "Cash through the bank",
    "NAME_TYPE_SUITE": "Family",
    "NAME_CLIENT_TYPE": "Repeater",
    "NAME_GOODS_CATEGORY": "Mobile",
    "CHANNEL_TYPE": "Country-wide",
    "SELLERPLACE_AREA": 42,
    "NAME_SELLER_INDUSTRY": "Connectivity",
    "CNT_PAYMENT": 4.0,
    "PRODUCT_COMBINATION": "POS mobile with interest",
    "NFLAG_INSURED_ON_APPROVAL": 1.0,
    "Weekday": "THURSDAY",
    "Hour": 2,
}


def test_fee_model(data):
    t0 = time.time_ns() // 1_000_000
    response = client.post("/consumer_loan_fee", json=data)
    t1 = time.time_ns() // 1_000_000
    print(f"Time taken: {t1-t0} ms")
    print(response.status_code)
    print(response.json())


test_fee_model(data=test_data)
