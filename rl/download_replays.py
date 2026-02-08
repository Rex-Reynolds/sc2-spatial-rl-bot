#!/usr/bin/env python3
"""
Automatically download StarCraft II replays from spawningtool.com

Usage:
    python rl/download_replays.py --player Maru --count 50
    python rl/download_replays.py --player Clem --count 30
"""

import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from urllib.parse import urljoin

def download_replays_from_spawningtool(player_name: str, max_replays: int, output_dir: str):
    """
    Scrape and download replays from spawningtool.com for a specific player.

    Args:
        player_name: Pro player name (e.g., "Maru", "Clem")
        max_replays: Maximum number of replays to download
        output_dir: Directory to save replays
    """
    base_url = "https://lotv.spawningtool.com"
    search_url = f"{base_url}/replays/?p={player_name}"

    print(f"Downloading replays for {player_name}")
    print(f"Target: {max_replays} replays")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1

    while downloaded < max_replays:
        print(f"\nFetching page {page}...")

        # Get page with replay listings
        if page == 1:
            page_url = search_url
        else:
            page_url = f"{search_url}&page={page}"

        try:
            response = requests.get(page_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all replay links on the page
        # spawningtool uses links like /replays/12345/
        replay_links = soup.find_all('a', href=lambda x: x and '/replays/' in x and x.count('/') >= 3)

        if not replay_links:
            print("No more replays found")
            break

        print(f"Found {len(replay_links)} replay links on this page")

        for link in replay_links:
            if downloaded >= max_replays:
                break

            replay_url = urljoin(base_url, link['href'])

            # Skip if not a replay detail page
            if not replay_url.endswith('/'):
                continue

            print(f"\n[{downloaded + 1}/{max_replays}] Processing: {replay_url}")

            try:
                # Get the replay detail page
                detail_response = requests.get(replay_url, timeout=10)
                detail_response.raise_for_status()
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')

                # Find the download link
                download_link = detail_soup.find('a', string='Download Replay')
                if not download_link:
                    # Try alternative text
                    download_link = detail_soup.find('a', href=lambda x: x and 'download' in x.lower())

                if not download_link:
                    print("  ⚠️  No download link found, skipping")
                    continue

                download_url = urljoin(base_url, download_link['href'])

                # Extract replay ID for filename
                replay_id = replay_url.strip('/').split('/')[-1]
                filename = f"{player_name}_{replay_id}.SC2Replay"
                output_file = output_path / filename

                # Skip if already downloaded
                if output_file.exists():
                    print(f"  ✓ Already exists: {filename}")
                    downloaded += 1
                    continue

                # Download the replay
                print(f"  Downloading: {filename}")
                replay_response = requests.get(download_url, timeout=30)
                replay_response.raise_for_status()

                # Save to file
                with open(output_file, 'wb') as f:
                    f.write(replay_response.content)

                file_size = len(replay_response.content) / 1024  # KB
                print(f"  ✓ Downloaded: {filename} ({file_size:.1f} KB)")

                downloaded += 1

                # Be nice to the server - small delay between downloads
                time.sleep(1)

            except requests.RequestException as e:
                print(f"  ✗ Error downloading: {e}")
                continue
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                continue

        # Move to next page
        page += 1

        # Be nice to the server - delay between pages
        time.sleep(2)

    print("\n" + "=" * 60)
    print(f"✓ Download complete!")
    print(f"Total replays downloaded: {downloaded}")
    print(f"Saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically download StarCraft II replays"
    )
    parser.add_argument(
        "--player",
        default="Maru",
        help="Pro player name (default: Maru)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of replays to download (default: 50)"
    )
    parser.add_argument(
        "--output",
        default="rl/data/replays/terran_pro",
        help="Output directory (default: rl/data/replays/terran_pro)"
    )

    args = parser.parse_args()

    print("StarCraft II Replay Downloader")
    print()

    # Check if beautifulsoup4 is installed
    try:
        import bs4
    except ImportError:
        print("Error: beautifulsoup4 not installed")
        print("Install with: pip install beautifulsoup4")
        return

    # Download replays
    download_replays_from_spawningtool(
        player_name=args.player,
        max_replays=args.count,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
