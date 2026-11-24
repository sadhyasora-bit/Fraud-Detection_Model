
"""Test Gemini Interactions API for CreditGuard AI."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.6-flash"


def main():
    if not API_KEY:
        print("CREDITGUARD AI: FAILED")
        print("GEMINI_API_KEY tidak ditemukan.")
        return

    client = None

    try:
        print(f"Testing model: {MODEL}", flush=True)
        print("API: Interactions", flush=True)

        client = genai.Client(api_key=API_KEY)

        print("Sending request to Gemini...", flush=True)
        start_time = time.perf_counter()

        interaction = client.interactions.create(
            model=MODEL,
            input="Reply with exactly: CREDITGUARD CONNECTION OK",
            timeout=30,
            automatic_continuation=False,
        )

        elapsed = time.perf_counter() - start_time

        if not interaction.output_text:
            raise RuntimeError("Gemini tidak menghasilkan output teks.")

        print("\nCREDITGUARD AI CONNECTION: SUCCESS")
        print(f"Response time: {elapsed:.2f} seconds")
        print("Model:", MODEL)
        print("Response:", interaction.output_text)

    except Exception as error:
        print("\nCREDITGUARD AI CONNECTION: FAILED")
        print(f"Error type: {type(error).__name__}")
        print("Error:", error)

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
