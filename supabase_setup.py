import os
from supabase import create_client

# Initialize client
url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')
supabase = create_client(url, key)

# Create bucket
try:
    supabase.storage.create_bucket(
        "reports",
        options={"public": True}
    )
    print("Bucket created successfully!")
except Exception as e:
    print(f"Error creating bucket: {e}")