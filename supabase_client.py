import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase = None
if url and key:
    supabase: Client = create_client(url, key)


def save_prediction(symptoms, predictions):
    if not supabase:
        return

    data = {
        "symptoms": symptoms,
        "prediction1": predictions[0]["disease"] if len(predictions) > 0 else None,
        "confidence1": predictions[0]["confidence"] if len(predictions) > 0 else None,
        "prediction2": predictions[1]["disease"] if len(predictions) > 1 else None,
        "confidence2": predictions[1]["confidence"] if len(predictions) > 1 else None,
        "prediction3": predictions[2]["disease"] if len(predictions) > 2 else None,
        "confidence3": predictions[2]["confidence"] if len(predictions) > 2 else None,
    }

    response = supabase.table("predictions").insert(data).execute()

    if hasattr(response, 'error') and response.error:
        print(f"Supabase error: {response.error}")