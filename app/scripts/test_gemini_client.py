
import time

from src.agent.llm.gemini_client import generate_response


def main():
    print("Testing CreditGuard AI Gemini Client...", flush=True)

    start = time.perf_counter()

    try:
        print("Sending request to Gemini...", flush=True)

        response = generate_response(
            "Reply with exactly: CREDITGUARD CLIENT OK"
        )

        elapsed = time.perf_counter() - start

        print("\nGEMINI CLIENT: SUCCESS")
        print(f"Response time: {elapsed:.2f} seconds")
        print("Response:", response)

    except Exception as error:
        elapsed = time.perf_counter() - start

        print("\nGEMINI CLIENT: FAILED")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print("Error type:", type(error).__name__)
        print("Error:", error)


if __name__ == "__main__":
    main()
