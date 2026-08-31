from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils.io import save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a prediction-error safety margin to required mask iterations."
    )
    parser.add_argument(
        "--input-csv", default="data/loci_soft_cldice_required_iterations.csv"
    )
    parser.add_argument(
        "--output-csv", default="data/loci_soft_cldice_training_iterations.csv"
    )
    parser.add_argument("--margin-iterations", type=int, default=10)
    parser.add_argument(
        "--round-up-to",
        type=int,
        default=10,
        help="Round adjusted values up to this iteration multiple.",
    )
    parser.add_argument("--minimum-iterations", type=int, default=0)
    parser.add_argument("--maximum-iterations", type=int, default=None)
    return parser.parse_args()


def add_iteration_margin(
    input_csv: str | Path,
    *,
    margin_iterations: int,
    round_up_to: int = 10,
    minimum_iterations: int = 0,
    maximum_iterations: int | None = None,
) -> list[dict[str, str | int]]:
    if margin_iterations < 0 or minimum_iterations < 0:
        raise ValueError("Iteration margin and minimum must be non-negative.")
    if round_up_to <= 0:
        raise ValueError("round_up_to must be positive.")
    if maximum_iterations is not None and maximum_iterations < minimum_iterations:
        raise ValueError("maximum_iterations must be at least minimum_iterations.")
    with Path(input_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "required_iterations" not in set(reader.fieldnames or []):
            raise ValueError("Input CSV must contain required_iterations.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    adjusted: list[dict[str, str | int]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            required = int(row["required_iterations"])
        except ValueError as error:
            raise ValueError(
                f"Invalid required_iterations at {input_csv}:{line_number}."
            ) from error
        if required < 0:
            raise ValueError(
                f"required_iterations must be non-negative at {input_csv}:{line_number}."
            )
        training = max(minimum_iterations, required + margin_iterations)
        training = ((training + round_up_to - 1) // round_up_to) * round_up_to
        if maximum_iterations is not None:
            training = min(training, maximum_iterations)
        adjusted.append(
            {
                **row,
                "margin_iterations": margin_iterations,
                "round_up_to": round_up_to,
                "training_iterations": training,
            }
        )
    return adjusted


def main() -> None:
    args = parse_args()
    rows = add_iteration_margin(
        args.input_csv,
        margin_iterations=args.margin_iterations,
        round_up_to=args.round_up_to,
        minimum_iterations=args.minimum_iterations,
        maximum_iterations=args.maximum_iterations,
    )
    output_path = Path(args.output_csv)
    save_csv(output_path, rows)
    print(f"Wrote {len(rows)} adjusted iteration rows to {output_path}")


if __name__ == "__main__":
    main()
