"""
Prompt templates for specific competition types
"""
STORE_SALES_TEMPLATE = """
You are solving the Store Sales - Time Series Forecasting competition on Kaggle.

Competition Details:
- Goal: Forecast store sales using time series forecasting
- Data: Multiple stores (1-54), various product families, date range 2013-2017
- Target: 'sales' column
- Evaluation: Root Mean Squared Logarithmic Error (RMSLE)
- Submission: CSV with 'id' and 'sales' columns

Key Considerations:
1. Time series nature: Use appropriate time-based validation
2. Multiple hierarchies: Store-level and product family-level patterns
3. External factors: Oil prices, holidays, promotions
4. Seasonality: Weekly (weekends), yearly (holidays)
5. Scale: Large dataset (~3M training rows)

Required Output:
- Complete Python script that generates submission.csv
- Must handle all data files: train, test, stores, oil, holidays, transactions
- Must create features like: day_of_week, month, year, is_weekend, is_holiday
- Must include lag features and rolling statistics
- Must use appropriate model (XGBoost, LightGBM, or similar)
- Must generate predictions for all test set ids (28512 records)

Generate the complete solution code:
```python
"""

def get_competition_specific_template(competition_name, problem_description):
    """Get appropriate template based on competition"""
    if "store-sales" in competition_name.lower():
        return STORE_SALES_TEMPLATE
    else:
        return None
