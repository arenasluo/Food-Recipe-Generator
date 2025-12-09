#!/usr/bin/env python3
"""
Upload the trained Qwen LoRA adapter to Hugging Face Hub.
"""
import os
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "arenasluo/qwen-recipe-lora"
LOCAL_MODEL_PATH = "./qwen_recipe_model_final"

# Files to upload (only LoRA adapter and tokenizer, not full base model)
FILES_TO_UPLOAD = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "README.md",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json"
]

def main():
    print(f"Uploading LoRA adapter to {REPO_ID}...")

    # Initialize API
    api = HfApi()

    # Check if logged in
    try:
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face!")
        print("Please run: huggingface-cli login")
        return

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload files
    print("\nUploading files...")
    for filename in FILES_TO_UPLOAD:
        file_path = os.path.join(LOCAL_MODEL_PATH, filename)

        if not os.path.exists(file_path):
            print(f"⚠️  Skipping {filename} (not found)")
            continue

        try:
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ❌ Error uploading {filename}: {e}")

    print(f"\n✓ Upload complete!")
    print(f"View your model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
