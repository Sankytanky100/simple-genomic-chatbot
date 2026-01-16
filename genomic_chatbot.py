#!/usr/bin/env python3
"""Genomic chatbot utilities and CLI.

This module provides a local, repeatable workflow for:
1) Crawling a knowledge source with Firecrawl and saving it as Markdown.
2) Starting a SambaNova-backed chat agent that uses the saved knowledge.
3) Exploring ENCODE public data from AWS S3.

Each CLI command is designed to be run independently, so users can mix and
match the steps in different environments (local machines, servers, etc.).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_PATH = ROOT_DIR / "local_data" / "sambanova_announcement.md"


@dataclass(frozen=True)
class EncodeDownloadResult:
    """Container for a downloaded ENCODE object."""

    object_key: str
    destination: Path


def require_package(package_name: str, install_hint: str) -> None:
    """Ensure an optional dependency is installed."""
    if importlib.util.find_spec(package_name) is None:
        raise RuntimeError(
            f"Missing optional dependency '{package_name}'. Install with: {install_hint}"
        )


def ensure_local_data_dir() -> Path:
    """Create and return the local data directory."""
    data_dir = ROOT_DIR / "local_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def build_firecrawl_client():
    """Build a Firecrawl client for the installed package version."""
    require_package("firecrawl", "pip install firecrawl-py")

    try:
        from firecrawl import FirecrawlApp

        return FirecrawlApp()
    except ImportError:
        from firecrawl import Firecrawl

        return Firecrawl()


def fetch_knowledge_with_firecrawl(url: str, output_path: Path) -> Path:
    """Crawl a URL with Firecrawl and save the Markdown output."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is not set in the environment.")

    firecrawl = build_firecrawl_client()
    response = firecrawl.crawl(url=url)
    if response.get("status") != "success":
        raise RuntimeError(f"Firecrawl failed: {response}")

    markdown = response["data"][0]["markdown"]
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def load_knowledge(knowledge_path: Path) -> str:
    """Load Markdown knowledge from disk."""
    if not knowledge_path.exists():
        raise FileNotFoundError(
            f"Knowledge file not found at {knowledge_path}. Run the crawl command first."
        )
    return knowledge_path.read_text(encoding="utf-8")


def start_chat(knowledge_path: Path) -> None:
    """Start an interactive SambaNova chat agent session."""
    require_package("camel", "pip install \"camel-ai[all]==0.2.11\"")
    from camel.agents import ChatAgent
    from camel.configs import SambaCloudAPIConfig
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    api_key = os.environ.get("SAMBA_API_KEY")
    if not api_key:
        raise RuntimeError("SAMBA_API_KEY is not set in the environment.")

    knowledge = load_knowledge(knowledge_path)
    qwen_model = ModelFactory.create(
        model_platform=ModelPlatformType.SAMBA,
        model_type="Qwen2.5-Coder-328-Instruct",
        model_config_dict=SambaCloudAPIConfig(max_tokens=500).as_dict(),
    )
    chat_agent = ChatAgent(
        system_message="You're a helpful assistant",
        message_window_size=20,
        model=qwen_model,
    )

    knowledge_message = BaseMessage.make_user_message(
        role_name="User",
        content=(
            "Use the following knowledge base as context for all answers. "
            f"Knowledge base:\n\n{knowledge}"
        ),
    )
    chat_agent.step(knowledge_message)

    print("Start chatting! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Ending conversation.")
            break
        assistant_response = chat_agent.step(user_input)
        print("Assistant:", assistant_response.msgs[0].content)


def create_unsigned_s3_client():
    """Create an unsigned S3 client for the public ENCODE bucket."""
    require_package("boto3", "pip install boto3 botocore")
    import boto3
    import botocore

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=botocore.client.Config(signature_version=botocore.UNSIGNED),
    )


def list_encode_objects(limit: int = 20) -> List[str]:
    """List object keys in the ENCODE public bucket."""
    s3_client = create_unsigned_s3_client()
    response = s3_client.list_objects_v2(Bucket="encode-public")
    keys = [obj["Key"] for obj in response.get("Contents", [])]
    return keys[:limit]


def download_encode_object(object_key: str, destination: Path) -> EncodeDownloadResult:
    """Download and optionally decompress an ENCODE object."""
    s3_client = create_unsigned_s3_client()
    response = s3_client.get_object(Bucket="encode-public", Key=object_key)
    data = response["Body"].read()

    if object_key.endswith(".gz"):
        import gzip
        import io

        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
            decompressed = gz.read()
        destination.write_bytes(decompressed)
    else:
        destination.write_bytes(data)

    return EncodeDownloadResult(object_key=object_key, destination=destination)


def read_urls_from_file(file_path: Path) -> List[str]:
    """Read URL lines from a file and return cleaned entries."""
    urls = [line.strip().strip('"') for line in file_path.read_text().splitlines()]
    return [url for url in urls if url]


def download_metadata(urls: Iterable[str], destination: Path) -> Path:
    """Download the ENCODE metadata file from the first URL."""
    urls_list = list(urls)
    if not urls_list:
        raise ValueError("No URLs provided to download metadata.")

    metadata_url = urls_list[0]
    response = requests.get(metadata_url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_sample_data(urls: Iterable[str], destination_dir: Path) -> Path | None:
    """Download the sample data file referenced as the second URL."""
    urls_list = list(urls)
    if len(urls_list) < 2:
        return None
    sample_url = urls_list[1]
    response = requests.get(sample_url, timeout=60)
    response.raise_for_status()
    file_name = sample_url.split("/@@download/")[-1]
    destination = destination_dir / file_name
    destination.write_bytes(response.content)
    return destination


def describe_metadata(metadata_path: Path, columns: List[str]) -> None:
    """Print preview and columns from a metadata TSV file."""
    require_package("pandas", "pip install pandas")
    import pandas as pd

    metadata_df = pd.read_csv(metadata_path, sep="\t")
    print("Metadata preview:")
    print(metadata_df.head())
    print("Metadata columns:")
    print(metadata_df.columns.tolist())
    if columns:
        print("Selected columns:")
        print(metadata_df[columns].head(5))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Genomic chatbot CLI utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl-knowledge", help="Fetch knowledge using Firecrawl")
    crawl_parser.add_argument("--url", required=True, help="URL to crawl")
    crawl_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_KNOWLEDGE_PATH,
        help="Path to save markdown knowledge",
    )

    chat_parser = subparsers.add_parser("chat", help="Start the SambaNova-powered chat")
    chat_parser.add_argument(
        "--knowledge",
        type=Path,
        default=DEFAULT_KNOWLEDGE_PATH,
        help="Path to knowledge markdown",
    )

    encode_list = subparsers.add_parser("encode-list", help="List ENCODE public objects")
    encode_list.add_argument("--limit", type=int, default=20)

    encode_download = subparsers.add_parser("encode-download", help="Download an ENCODE object")
    encode_download.add_argument("object_key", help="Object key in the ENCODE bucket")
    encode_download.add_argument(
        "--destination",
        type=Path,
        default=ROOT_DIR / "local_data" / "encode_object.txt",
    )

    encode_metadata = subparsers.add_parser(
        "encode-metadata", help="Download metadata and sample file from a files.txt list"
    )
    encode_metadata.add_argument("files_list", type=Path)
    encode_metadata.add_argument(
        "--metadata-out",
        type=Path,
        default=ROOT_DIR / "local_data" / "metadata.tsv",
    )
    encode_metadata.add_argument(
        "--columns",
        nargs="*",
        default=[],
        help="Columns to display from metadata",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "crawl-knowledge":
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path = fetch_knowledge_with_firecrawl(args.url, output_path)
        print(f"Saved knowledge to {saved_path}")
        return

    if args.command == "chat":
        start_chat(args.knowledge)
        return

    if args.command == "encode-list":
        for key in list_encode_objects(limit=args.limit):
            print(key)
        return

    if args.command == "encode-download":
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        result = download_encode_object(args.object_key, args.destination)
        print(f"Downloaded {result.object_key} to {result.destination}")
        return

    if args.command == "encode-metadata":
        ensure_local_data_dir()
        urls = read_urls_from_file(args.files_list)
        metadata_path = download_metadata(urls, args.metadata_out)
        sample_path = download_sample_data(urls, args.metadata_out.parent)
        print(f"Metadata saved to {metadata_path}")
        if sample_path:
            print(f"Sample data saved to {sample_path}")
        describe_metadata(metadata_path, args.columns)
        return


if __name__ == "__main__":
    main()
