
from src.agent.llm.gemini_client import get_gemini_client

try:
    client = get_gemini_client()

    print("=" * 55)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 55)

    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []

        if "generateContent" in actions:
            print(f"Model: {model.name}")
            print(f"Supported actions: {actions}")
            print("-" * 55)

except Exception as error:
    print("FAILED TO RETRIEVE MODELS")
    print(f"Error: {error}")
