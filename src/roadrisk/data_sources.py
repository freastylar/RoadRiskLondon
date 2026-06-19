from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen

from roadrisk.config import DEFAULT_LATEST_FINAL_YEAR, Mode, PipelineConfig

OPEN_DATA_PAGE = "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data"
STATISTICS_COLLECTION = "https://www.gov.uk/government/collections/road-safety-statistics"
GUIDANCE_PAGE = "https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance"

TABLES = ("collisions", "vehicles", "casualties")


@dataclass(frozen=True)
class SourceFile:
    table: str
    label: str
    url: str
    filename: str
    start_year: int | None
    end_year: int | None
    final: bool
    provisional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise_table(label: str) -> str | None:
    text = label.lower()
    if "collision" in text:
        return "collisions"
    if "vehicle" in text:
        return "vehicles"
    if "casualt" in text:
        return "casualties"
    return None


def _filename_from_url(url: str) -> str:
    return Path(url.split("?", 1)[0]).name


def _extract_years(label: str) -> tuple[int | None, int | None]:
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", label)]
    if "last 5 years" in label.lower():
        return DEFAULT_LATEST_FINAL_YEAR - 4, DEFAULT_LATEST_FINAL_YEAR
    if "latest published year" in label.lower():
        return 1979, DEFAULT_LATEST_FINAL_YEAR
    if len(years) == 1:
        return years[0], years[0]
    if len(years) >= 2:
        return min(years), max(years)
    return None, None


def discover_open_data_sources(page_url: str = OPEN_DATA_PAGE) -> list[SourceFile]:
    request = Request(page_url, headers={"User-Agent": "RoadRiskLondon/0.1"})
    with urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    anchors = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S)
    sources: list[SourceFile] = []
    for href, label_html in anchors:
        label = re.sub(r"<[^>]+>", " ", label_html)
        label = re.sub(r"\s+", " ", unescape(label)).strip()
        table = _normalise_table(label)
        if table is None or "data.dft.gov.uk" not in href:
            continue
        provisional = "provisional" in label.lower()
        start_year, end_year = _extract_years(label)
        sources.append(
            SourceFile(
                table=table,
                label=label,
                url=href,
                filename=_filename_from_url(href),
                start_year=start_year,
                end_year=end_year,
                final=not provisional,
                provisional=provisional,
            )
        )
    return sources


def _covers(source: SourceFile, start_year: int, end_year: int) -> bool:
    if source.start_year is None or source.end_year is None:
        return False
    return source.start_year <= start_year and source.end_year >= end_year


def _annual_sources(sources: Iterable[SourceFile], table: str, years: Iterable[int]) -> list[SourceFile]:
    selected: list[SourceFile] = []
    for year in years:
        matches = [
            source
            for source in sources
            if source.table == table
            and source.final
            and not source.provisional
            and source.start_year == year
            and source.end_year == year
        ]
        if matches:
            selected.append(matches[0])
    return selected


def select_sources(config: PipelineConfig, sources: list[SourceFile]) -> list[SourceFile]:
    selected: list[SourceFile] = []
    for table in TABLES:
        candidates = [source for source in sources if source.table == table and source.final and not source.provisional]
        if config.mode == Mode.TRENDS:
            full = [
                source
                for source in candidates
                if source.start_year == 1979 and source.end_year == config.latest_final_year
            ]
            if full:
                selected.append(full[0])
                continue
        covering = [source for source in candidates if _covers(source, config.year_range.start_year, config.year_range.end_year)]
        latest_five = [source for source in covering if "last 5 years" in source.label.lower()]
        if latest_five and config.year_range.start_year >= DEFAULT_LATEST_FINAL_YEAR - 4:
            selected.append(latest_five[0])
            continue
        annual = _annual_sources(candidates, table, config.year_range.years)
        if annual:
            selected.extend(annual)
            continue
        if covering:
            selected.append(covering[0])
            continue
        msg = f"No final DfT source found for {table} covering {config.year_range.label}"
        raise RuntimeError(msg)
    return selected


def documentation_sources() -> list[dict[str, str]]:
    return [
        {"label": "Road safety open data", "url": OPEN_DATA_PAGE},
        {"label": "Road safety statistics", "url": STATISTICS_COLLECTION},
        {"label": "Road safety statistics guidance", "url": GUIDANCE_PAGE},
    ]
