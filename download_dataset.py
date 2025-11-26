import subprocess
import sys

dataset = "pes12017000148/food-ingredients-and-recipe-dataset-with-images"
dest = "./dataset"

print(f"Downloading {dataset}...")

try:
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", dest, "--unzip"], check=True)
    print(f"[OK] Dataset ready in {dest}")
except FileNotFoundError:
    print("[ERROR] Kaggle CLI not found. Install: pip install kaggle")
    print("Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)

