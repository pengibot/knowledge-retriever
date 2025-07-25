# check_api.py
import os
from huggingface_hub import InferenceClient

# --- Configuration ---
# Make sure your secrets file exists at .streamlit/secrets.toml
# with your token: HUGGINGFACEHUB_API_TOKEN = "hf_..."
#
# This script reads the token the same way Streamlit does.
# You can also set it manually for testing:
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_YourNewWriteToken"

# The model and task we want to test
repo_id = "google/flan-t5-base"
task = "text2text-generation"
prompt = "Translate to German: My name is Arthur"

# --- Main Test ---
print(f"Attempting to call model: {repo_id}")
print("="*30)

try:
    # 1. Load the token from secrets (requires a bit of extra code)
    # This part simulates how Streamlit loads secrets
    from toml import load as toml_load
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        secrets = toml_load(secrets_path)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
        print("Token loaded from secrets.toml.")
    else:
        print("secrets.toml not found. Make sure your token is set as an environment variable.")

    # 2. Create the client
    client = InferenceClient() # The client automatically uses the token from the environment

    # 3. Check model status first for better debugging
    print(f"\nChecking status for {repo_id}...")
    status = client.get_model_status(repo_id)
    print(f"Model Status: {status}")
    if not status.loaded:
        print(f"Model is not loaded. Current state: {status.state}. This might take a while.")
        print("The script will still attempt to call it, as this can trigger a load.")

    # 4. Make the inference call
    print(f"\nAttempting to generate text with prompt: '{prompt}'")
    response_bytes = client.post(
        json={"inputs": prompt},
        model=repo_id,
        task=task,
    )

    print("\n✅ SUCCESS! The API call worked.")
    print("Response:", response_bytes.decode())

except Exception as e:
    print(f"\n❌ FAILED. The API call did not work.")
    print("Error Type:", type(e).__name__)
    print("Error Details:", e)