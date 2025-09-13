import argparse
from summarizer.models import download_model


def main():
    p = argparse.ArgumentParser(description="Download summarization models locally")
    p.add_argument("--size", choices=["small", "medium", "large"], required=True)
    p.add_argument("--dir", default="models", help="Target directory for local models")
    args = p.parse_args()

    local_dir = download_model(args.size, args.dir)
    print("Downloaded to:", local_dir)


if __name__ == "__main__":
    main()

