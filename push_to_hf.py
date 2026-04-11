import os
from huggingface_hub import HfApi, login

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("Please set HF_TOKEN environment variable")

login(token=token)

api = HfApi()
user_info = api.whoami()
username = user_info["name"]

repo_name = "smartgrid-optima"
repo_id = f"{username}/{repo_name}"

print(f"User: {username}")
print(f"Creating/getting Space: {repo_id}...")

try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )
    print("Space exists or created successfully.")
except Exception as e:
    print(f"Error creating space: {e}")

local_dir = "."
print(f"Uploading files from {local_dir} to {repo_id}...")
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        "__pycache__", "*.pyc", ".git", ".venv", "push_to_hf.py", "venv", "*.txt", "*.log"
    ]
)

print(f"Done! Space URL: https://huggingface.co/spaces/{repo_id}")
