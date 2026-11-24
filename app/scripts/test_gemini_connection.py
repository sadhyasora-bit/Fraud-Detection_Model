from src.agent.llm.gemini_client import generate_response


if __name__ == "__main__":
    prompt = """
    You are CreditGuard AI Assistant, an AI assistant
    integrated into the CreditGuard AI credit card
    fraud detection application.

    Your responsibilities include:
    - Explaining fraud detection results.
    - Assisting transaction risk analysis.
    - Supporting fraud investigation.
    - Providing evidence-based recommendations.

    You must never invent transaction data,
    modify machine learning predictions,
    or override deterministic risk policies.

    Introduce yourself briefly in Bahasa Indonesia.
    """

    try:
        result = generate_response(prompt)

        print("=" * 50)
        print("CREDITGUARD AI CONNECTION: SUCCESS")
        print("=" * 50)
        print(result)

    except Exception as error:
        print("CREDITGUARD AI CONNECTION: FAILED")
        print(f"Error: {error}")