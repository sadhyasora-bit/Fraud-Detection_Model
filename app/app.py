"""CreditGuard AI application entry point and recovered page navigation."""
from streamlit import Page, navigation

workspace = [
    Page("pages/1_Dashboard_Overview.py", title="Dashboard Overview", default=True),
    Page("pages/2_Transaction_Monitoring.py", title="Transaction Monitoring"),
    Page("pages/3_Fraud_Investigation.py", title="Fraud Investigation"),
    Page("pages/4_Agentic_AI_Center.py", title="Agentic AI Center"),
]
analytics = [
    Page("pages/5_Model_Performance.py", title="Model Performance"),
    Page("pages/6_Experiment_Lab.py", title="Experiment Lab"),
    Page("pages/7_Research_Evaluation.py", title="Research Evaluation"),
    Page("pages/8_Settings_Dataset.py", title="Settings & Dataset"),
]

navigation({"Workspace": workspace, "Analytics & Research": analytics}).run()

