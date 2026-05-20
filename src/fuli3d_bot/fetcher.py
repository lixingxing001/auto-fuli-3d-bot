from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from .models import Draw
from .repository import parse_date, write_draws


CWL_DRAW_NOTICE_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"


@dataclass(frozen=True)
class FetchResult:
    draws: list[Draw]
    total_remote: int
    pages_fetched: int
    source_url: str


def build_cwl_url(page_no: int, page_size: int) -> str:
    params = {
        "name": "3d",
        "issueCount": "",
        "issueStart": "",
        "issueEnd": "",
        "dayStart": "",
        "dayEnd": "",
        "pageNo": str(page_no),
        "pageSize": str(page_size),
        "week": "",
        "systemType": "PC",
    }
    return f"{CWL_DRAW_NOTICE_URL}?{urlencode(params)}"


def _load_json(url: str, timeout: int, opener: Any) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.cwl.gov.cn/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
    )
    with opener.open(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("unexpected CWL response shape")
    return data


def _parse_cwl_date(value: str) -> date | None:
    raw = value.strip()
    if "(" in raw:
        raw = raw.split("(", 1)[0]
    return parse_date(raw)


def _parse_cwl_row(row: dict[str, Any]) -> Draw:
    issue = str(row.get("code", "")).strip()
    red = str(row.get("red", "")).strip()
    number = "".join(part.strip() for part in red.split(","))
    if not issue:
        raise ValueError(f"missing issue in row: {row!r}")
    if len(number) != 3 or not number.isdigit():
        raise ValueError(f"invalid red number for issue {issue}: {red!r}")
    return Draw(issue=issue, draw_date=_parse_cwl_date(str(row.get("date", ""))), number=number)


def fetch_cwl_draws(
    page_size: int = 100,
    max_pages: int | None = None,
    timeout: int = 20,
) -> FetchResult:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")

    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    first_url = build_cwl_url(page_no=1, page_size=page_size)
    first_page = _load_json(first_url, timeout=timeout, opener=opener)
    if first_page.get("state") != 0:
        raise ValueError(f"CWL query failed: {first_page.get('message')}")

    total_remote = int(first_page.get("total") or 0)
    page_num = int(first_page.get("pageNum") or 1)
    pages_to_fetch = min(page_num, max_pages) if max_pages else page_num

    rows: list[dict[str, Any]] = []
    first_rows = first_page.get("result") or []
    if not isinstance(first_rows, list):
        raise ValueError("unexpected CWL result field")
    rows.extend(first_rows)

    for page_no in range(2, pages_to_fetch + 1):
        page = _load_json(
            build_cwl_url(page_no=page_no, page_size=page_size),
            timeout=timeout,
            opener=opener,
        )
        if page.get("state") != 0:
            raise ValueError(f"CWL query failed on page {page_no}: {page.get('message')}")
        page_rows = page.get("result") or []
        if not isinstance(page_rows, list):
            raise ValueError(f"unexpected CWL result field on page {page_no}")
        rows.extend(page_rows)

    draws = [_parse_cwl_row(row) for row in rows]
    draws.sort(key=lambda item: item.issue)
    return FetchResult(
        draws=draws,
        total_remote=total_remote,
        pages_fetched=pages_to_fetch,
        source_url=first_url,
    )


def fetch_and_write(
    output_path: str | Path,
    page_size: int = 100,
    max_pages: int | None = None,
    limit: int | None = None,
    timeout: int = 20,
) -> FetchResult:
    result = fetch_cwl_draws(page_size=page_size, max_pages=max_pages, timeout=timeout)
    draws = result.draws[-limit:] if limit else result.draws
    write_draws(output_path, draws)
    return FetchResult(
        draws=draws,
        total_remote=result.total_remote,
        pages_fetched=result.pages_fetched,
        source_url=result.source_url,
    )
