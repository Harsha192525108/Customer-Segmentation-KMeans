# Customer Segmentation and Behavioural Pattern Analysis

A Streamlit application that uses **K-Means clustering** to group customers and analyse behavioural patterns for data-driven marketing insights.

## Project modules

1. **Customer Data Processing and Segmentation**
   - Upload a customer CSV or use the built-in sample dataset.
   - Select numeric features, handle incomplete records and standardise values.
   - Apply K-Means clustering and visualise the customer segments.

2. **Behavioural Pattern Analysis and Marketing Insights**
   - Compare average customer characteristics across segments.
   - Explore relationships between behavioural features.
   - View suggested marketing actions and download results as CSV files.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Deploy with Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app** and select this repository.
3. Select branch `main` and main file path `app.py`.
4. Click **Deploy**.

## Dataset format

Upload a CSV containing at least two numeric columns. Suggested fields include CustomerID, Age, Annual Income, Spending Score, Purchase Frequency, Average Order Value, Total Spend and Recency (days). Column names may differ; choose the features in the app.

The built-in dataset is synthetic and intended for demonstration. K-Means standardises the selected features before clustering.
