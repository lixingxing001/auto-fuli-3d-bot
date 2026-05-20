from __future__ import annotations

from dataclasses import dataclass, field

from .models import Draw


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    duplicate_issues: list[str] = field(default_factory=list)
    issue_gaps: list[str] = field(default_factory=list)
    date_gaps: list[str] = field(default_factory=list)
    out_of_order: bool = False

    @property
    def ok(self) -> bool:
        return not self.duplicate_issues and not self.out_of_order


def _issue_parts(issue: str) -> tuple[int, int] | None:
    if len(issue) < 5 or not issue.isdigit():
        return None
    return int(issue[:4]), int(issue[4:])


def validate_draws(draws: list[Draw]) -> ValidationReport:
    duplicate_issues: list[str] = []
    seen: set[str] = set()
    for draw in draws:
        if draw.issue in seen:
            duplicate_issues.append(draw.issue)
        seen.add(draw.issue)

    out_of_order = any(draws[index].issue > draws[index + 1].issue for index in range(len(draws) - 1))

    issue_gaps: list[str] = []
    for previous, current in zip(draws, draws[1:]):
        previous_parts = _issue_parts(previous.issue)
        current_parts = _issue_parts(current.issue)
        if previous_parts is None or current_parts is None:
            continue
        previous_year, previous_seq = previous_parts
        current_year, current_seq = current_parts
        if previous_year == current_year and current_seq != previous_seq + 1:
            issue_gaps.append(f"{previous.issue}->{current.issue}")

    date_gaps: list[str] = []
    for previous, current in zip(draws, draws[1:]):
        if previous.draw_date is None or current.draw_date is None:
            continue
        delta = (current.draw_date - previous.draw_date).days
        if delta > 1:
            date_gaps.append(f"{previous.issue}->{current.issue}:{delta}d")

    return ValidationReport(
        rows=len(draws),
        duplicate_issues=sorted(set(duplicate_issues)),
        issue_gaps=issue_gaps,
        date_gaps=date_gaps,
        out_of_order=out_of_order,
    )
