# 🛒 SMart — Customer Purchase Pattern Mining System

A Data Mining academic project that discovers customer purchase patterns and segments customers.

## Objectives
- Clean and prepare transaction data.
- Discover frequent product combinations using Apriori.
- Generate association rules using support, confidence and lift.
- Segment customers using K-Means clustering.
- Present results through an interactive Streamlit dashboard.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, mlxtend, Matplotlib, Streamlit.

## Project Pipeline
Raw Transactions → Data Cleaning → Apriori Association Mining
→ K-Means Customer Segmentation → Business Insights

## How to Run

### 1. Open the project in VS Code

### 2. Create/activate a virtual environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the dashboard
```bash
streamlit run app.py
```

## Algorithms

### Apriori
Finds frequent itemsets and association rules in transaction data.

### K-Means
Groups customers based on similar purchasing behavior.

## Key Metrics
- Support
- Confidence
- Lift
- Cluster-level customer behavior

## Important Academic Note
The included dataset is synthetic and was generated for demonstration/academic use. This keeps the project reproducible while allowing the algorithms to discover meaningful purchase associations.

## Suggested Presentation
1. Problem statement
2. What is Data Mining?
3. Dataset
4. Data preprocessing
5. Apriori and association rules
6. K-Means customer segmentation
7. Dashboard demo
8. Results and future scope
