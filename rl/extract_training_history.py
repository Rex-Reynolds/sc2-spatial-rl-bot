#!/usr/bin/env python3
"""
Extract training history from TensorBoard logs and display in human-readable format.

Converts binary TensorBoard event files into readable text/CSV.
"""

import argparse
from pathlib import Path
import struct


def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard event files and extract metrics."""
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    # Find all event files
    event_files = list(log_path.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return

    print(f"Found {len(event_files)} event file(s)")
    print("=" * 70)

    try:
        from tensorboard.backend.event_processing import event_accumulator

        for event_file in event_files:
            print(f"\nProcessing: {event_file}")

            # Load events
            ea = event_accumulator.EventAccumulator(str(event_file.parent))
            ea.Reload()

            # Get available tags
            tags = ea.Tags()

            if not tags['scalars']:
                print("  ⚠️  No scalar data found (file may be empty or incomplete)")
                continue

            print(f"  Found metrics: {', '.join(tags['scalars'])}")

            # Extract each metric
            for tag in tags['scalars']:
                events = ea.Scalars(tag)
                print(f"\n  {tag}:")
                for event in events[-5:]:  # Show last 5 values
                    print(f"    Step {event.step}: {event.value:.4f}")

    except ImportError:
        print("\n⚠️  TensorBoard not installed. Install with:")
        print("  pip install tensorboard")
        print("\nAttempting basic file inspection instead...")

        # Fallback: just show file sizes
        print("\nEvent files:")
        for event_file in event_files:
            size = event_file.stat().st_size
            print(f"  {event_file.name}: {size} bytes")

            if size == 88:
                print("    ⚠️  Empty (only header, no training data)")
            else:
                print(f"    ✓  Contains data")


def export_to_csv(log_dir, output_file):
    """Export training metrics to CSV."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import csv

        log_path = Path(log_dir)
        event_files = list(log_path.rglob("events.out.tfevents.*"))

        if not event_files:
            print(f"No event files found in {log_dir}")
            return

        # Collect all data
        all_data = []

        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(str(event_file.parent))
            ea.Reload()

            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                for event in events:
                    all_data.append({
                        'metric': tag,
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time,
                    })

        if not all_data:
            print("No data to export")
            return

        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'step', 'value', 'wall_time'])
            writer.writeheader()
            writer.writerows(all_data)

        print(f"✓ Exported {len(all_data)} data points to {output_file}")

    except ImportError:
        print("Error: TensorBoard not installed. Cannot export to CSV.")


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard training history")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="rl/logs/sc2_ppo",
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--csv",
        help="Export to CSV file"
    )

    args = parser.parse_args()

    if args.csv:
        export_to_csv(args.log_dir, args.csv)
    else:
        parse_tensorboard_logs(args.log_dir)


if __name__ == "__main__":
    main()
