"""Import previous application HouseholdRequest JSON submissions or inspect a history snapshot."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.uncertain_twin import HouseholdRequest
from greentwin.twin.application_history import DEFAULT_HISTORY, ApplicationHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("import", "inspect"))
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--file",
        type=Path,
        help="A request, list of requests, or {submissions:[{request,submitted_at,kind}]} export",
    )
    args = parser.parse_args()
    history = ApplicationHistory(args.history)
    try:
        if args.mode == "import":
            if not args.file:
                raise ValueError("import requires --file")
            data = json.loads(args.file.read_text())
            entries = data.get("submissions", [data]) if isinstance(data, dict) else data
            imported = 0
            for entry in entries:
                request = HouseholdRequest.model_validate(entry.get("request", entry))
                submitted = (
                    datetime.fromisoformat(entry["submitted_at"]) if entry.get("submitted_at") else None
                )
                imported += history.record(
                    request, submitted_at=submitted, kind=entry.get("kind", "baseline")
                )
            print(f"Recorded {imported} baseline submissions (latest revisions count once per household)")
        snapshot = history.snapshot(datetime.now(UTC))
        print(f"History: {history.path}; {len(snapshot)} distinct current household records")
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Application history failed: {exc}\n")


if __name__ == "__main__":
    main()
