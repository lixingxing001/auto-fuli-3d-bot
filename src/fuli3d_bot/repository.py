from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .models import Draw


REQUIRED_FIELDS = ("issue", "date", "number")


def parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unsupported date format: {value!r}")


def load_draws(path: str | Path) -> list[Draw]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"history file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required CSV fields: {', '.join(missing)}")

        draws = [
            Draw(
                issue=str(row["issue"]).strip(),
                draw_date=parse_date(row.get("date")),
                number=str(row["number"]).strip(),
            )
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    if not draws:
        raise ValueError(f"history file has no draw rows: {csv_path}")

    seen_issues: set[str] = set()
    duplicates: list[str] = []
    for draw in draws:
        if draw.issue in seen_issues:
            duplicates.append(draw.issue)
        seen_issues.add(draw.issue)
    if duplicates:
        raise ValueError(f"duplicate issue values: {', '.join(sorted(set(duplicates)))}")

    return draws


def write_draws(path: str | Path, draws: Iterable[Draw]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for draw in draws:
            writer.writerow(
                {
                    "issue": draw.issue,
                    "date": draw.draw_date.isoformat() if draw.draw_date else "",
                    "number": draw.number,
                }
            )

