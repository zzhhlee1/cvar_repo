#!/usr/bin/env python3
"""Fetch public air-cargo calibration data for the application MVP.

Sources:
- BTS TranStats T-100 Segment (All Carriers), monthly zip downloads.
- BTS Socrata AFF T-100 summary tables.
- FRED air freight/mail revenue ton-miles.
- IATA public air-cargo market-analysis page metadata.

The script intentionally fetches public aggregate/segment data only; it does
not assume access to paid rate feeds or proprietary booking logs.
"""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path

import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "data" / "sources"

TRANSTATS_T100_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx?"
    "QO_fu146_anzr=Nv4+Pn44vr45&gnoyr_VQ=FMG"
)

T100_FIELDS = [
    "DEPARTURES_SCHEDULED",
    "DEPARTURES_PERFORMED",
    "PAYLOAD",
    "FREIGHT",
    "MAIL",
    "DISTANCE",
    "UNIQUE_CARRIER",
    "UNIQUE_CARRIER_NAME",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "ORIGIN_COUNTRY",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "DEST_COUNTRY",
    "AIRCRAFT_TYPE",
    "AIRCRAFT_CONFIG",
    "YEAR",
    "MONTH",
    "CLASS",
    "DATA_SOURCE",
]

FRED_URLS = {
    "fred_air_revenue_ton_miles_sa.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AIRRTMFM",
    "fred_air_revenue_ton_miles_nsa.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AIRRTMFMD11",
}

SOCRATA_URLS = {
    "bts_aff_t100_segment_summary_by_carrier.csv": "https://data.transportation.gov/api/views/q4tb-tbff/rows.csv?accessType=DOWNLOAD",
    "bts_aff_t100_segment_summary_by_origin_airport.csv": "https://data.transportation.gov/api/views/r495-tyji/rows.csv?accessType=DOWNLOAD",
    "bts_aff_t100_segment_summary_by_year.csv": "https://data.transportation.gov/api/views/bu82-4pwz/rows.csv?accessType=DOWNLOAD",
}

SOURCE_PAGES = {
    "transtats_t100_segment_download_page.html": TRANSTATS_T100_URL,
    "iata_air_cargo_market_analysis_jan_2026.html": "https://www.iata.org/en/publications/economics/reports/air-cargo-market-analysis-january-2026/",
    "arcgis_t100_item.json": "https://www.arcgis.com/sharing/rest/content/items/17e9a793c7cf47c8b64dab92da55dfe5?f=json",
    "arcgis_t100_featureserver.json": "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/T100_Domestic_Market_and_Segment_Data/FeatureServer?f=json",
}


def ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)


def download(session: requests.Session, url: str, path: Path, *, verify: bool = True) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"skip existing {path}")
        return
    print(f"fetch {url} -> {path}")
    with session.get(url, stream=True, timeout=180, verify=verify) as resp:
        resp.raise_for_status()
        with path.open("wb") as fh:
            for chunk in resp.iter_content(1024 * 128):
                if chunk:
                    fh.write(chunk)
    print(f"  wrote {path.stat().st_size:,} bytes")


def hidden_value(html: str, name: str) -> str:
    m = re.search(r'name="%s"[^>]*value="([^"]*)"' % re.escape(name), html)
    if not m:
        raise RuntimeError(f"missing hidden field {name}")
    return m.group(1)


def fetch_t100_months(session: requests.Session, months: list[tuple[int, int]]) -> None:
    print("fetch TranStats T-100 form state")
    page = session.get(TRANSTATS_T100_URL, timeout=60, verify=False)
    page.raise_for_status()
    html = page.text
    (SOURCES / "transtats_t100_segment_download_page.html").write_text(html)
    hidden = {
        "__VIEWSTATE": hidden_value(html, "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": hidden_value(html, "__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION": hidden_value(html, "__EVENTVALIDATION"),
    }

    for year, month in months:
        out = RAW / f"bts_t100_segment_all_carriers_{year}_{month:02d}.zip"
        if out.exists() and out.stat().st_size > 0:
            print(f"skip existing {out}")
            continue
        data = dict(hidden)
        data.update(
            {
                "cboGeography": "All",
                "cboYear": str(year),
                "cboPeriod": str(month),
                "btnDownload": "Download",
            }
        )
        for field in T100_FIELDS:
            data[field] = "on"
        label = f"{year}-{month:02d} ({calendar.month_name[month]})"
        print(f"post TranStats T-100 {label}")
        with session.post(TRANSTATS_T100_URL, data=data, stream=True, timeout=180, verify=False) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "zip" not in ctype.lower():
                preview = resp.content[:500]
                raise RuntimeError(f"expected zip for {label}, got {ctype}: {preview!r}")
            with out.open("wb") as fh:
                for chunk in resp.iter_content(1024 * 128):
                    if chunk:
                        fh.write(chunk)
        print(f"  wrote {out.stat().st_size:,} bytes")


def default_months() -> list[tuple[int, int]]:
    return [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-t100", action="store_true", help="Skip TranStats monthly segment zips.")
    args = parser.parse_args()

    ensure_dirs()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for filename, url in FRED_URLS.items():
        download(session, url, RAW / filename)
    for filename, url in SOCRATA_URLS.items():
        download(session, url, RAW / filename)
    for filename, url in SOURCE_PAGES.items():
        download(session, url, SOURCES / filename, verify=False)

    if not args.skip_t100:
        fetch_t100_months(session, default_months())


if __name__ == "__main__":
    main()
