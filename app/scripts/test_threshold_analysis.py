
from src.agent.tools.threshold_analysis import (
    run_threshold_analysis,
)


test_cases = [
    ("Normal probability", 0.000004, 0.50),
    ("Fraud probability", 0.999780, 0.50),
    ("Boundary probability", 0.50, 0.50),
]

for name, probability, threshold in test_cases:

    result = run_threshold_analysis(
        fraud_probability=probability,
        threshold=threshold,
    )

    print(f"\n===== {name} =====")

    for key, value in result.items():
        print(f"{key}: {value}")
