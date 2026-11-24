
from src.agent.llm.gemini_client import get_gemini_client

client = get_gemini_client()

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
]

print("=" * 55)
print("FIRA - GEMINI MODEL DIAGNOSTIC")
print("=" * 55)

working_model = None

for model in MODELS:
    print(f"\nTesting: {model}")

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: FIRA CONNECTION OK",
        )

        if response.text:
            print("STATUS: SUCCESS")
            print("RESPONSE:", response.text)
            working_model = model
            break

        print("STATUS: EMPTY RESPONSE")

    except Exception as error:
        print("STATUS: FAILED")
        print("ERROR:", str(error)[:300])

print("\n" + "=" * 55)

if working_model:
    print("WORKING MODEL:", working_model)
    print("Set GEMINI_MODEL to this model in .env")
else:
    print("NO MODEL AVAILABLE")
    print("Check Gemini API service status and try again.")

print("=" * 55)
