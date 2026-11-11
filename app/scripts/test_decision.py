from src.agent.tools.decision import run_decision


test_cases = [
    {
        "name": "Low Risk",
        "risk_result": {
            "status": "success",
            "risk_level": "LOW",
        },
    },
    {
        "name": "Medium Risk",
        "risk_result": {
            "status": "success",
            "risk_level": "MEDIUM",
        },
    },
    {
        "name": "High Risk",
        "risk_result": {
            "status": "success",
            "risk_level": "HIGH",
        },
    },
]

for test in test_cases:

    result = run_decision(test["risk_result"])

    print(f"\n===== {test['name']} =====")

    for key, value in result.items():
        print(f"{key}: {value}")