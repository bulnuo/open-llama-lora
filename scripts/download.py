import argparse
from huggingface_hub import snapshot_download

def download_from_hub(repo_id, local_dir):
    local_dir = f"{local_dir}/{repo_id}"
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a repo snapshot from HuggingFace hub")
    parser.add_argument('--repo_id', type=str, help="The ID of the repository to download")
    parser.add_argument('--local_dir', type=str, help="The local directory to store the downloaded snapshot")
    args = parser.parse_args()

    download_from_hub(args.repo_id, args.local_dir)

