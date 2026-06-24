#!/usr/bin/env python3
"""Build lightweight calibration tables from fetched public air-cargo data."""

from __future__ import annotations

import csv
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


NUMERIC = {
    "DEPARTURES_SCHEDULED",
    "DEPARTURES_PERFORMED",
    "PAYLOAD",
    "FREIGHT",
    "MAIL",
    "DISTANCE",
    "YEAR",
    "MONTH",
}


def as_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def iter_t100_rows():
    for zip_path in sorted(RAW.glob("bts_t100_segment_all_carriers_*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if not names:
                continue
            with zf.open(names[0]) as fh:
                reader = csv.DictReader(line.decode("utf-8-sig") for line in fh)
                for row in reader:
                    row["_source_zip"] = zip_path.name
                    yield row


def build_segment_sample() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "bts_t100_segment_monthly_selected_fields.csv"
    rows = list(iter_t100_rows())
    if not rows:
        raise RuntimeError("no monthly T-100 zip files found; run fetch script first")
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows):,}")


def build_route_month_summary() -> None:
    groups: dict[tuple[str, str, int, int], dict[str, float]] = defaultdict(
        lambda: {
            "departures_performed": 0.0,
            "payload_lbs": 0.0,
            "freight_lbs": 0.0,
            "mail_lbs": 0.0,
            "distance_miles_weighted": 0.0,
        }
    )
    meta: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in iter_t100_rows():
        year = int(float(row["YEAR"]))
        month = int(float(row["MONTH"]))
        key = (row["ORIGIN"], row["DEST"], year, month)
        dep = as_float(row["DEPARTURES_PERFORMED"])
        payload = as_float(row["PAYLOAD"])
        freight = as_float(row["FREIGHT"])
        mail = as_float(row["MAIL"])
        dist = as_float(row["DISTANCE"])
        groups[key]["departures_performed"] += dep
        groups[key]["payload_lbs"] += payload
        groups[key]["freight_lbs"] += freight
        groups[key]["mail_lbs"] += mail
        groups[key]["distance_miles_weighted"] += dist * max(dep, 1.0)
        meta.setdefault(
            key,
            {
                "origin_city": row.get("ORIGIN_CITY_NAME", ""),
                "dest_city": row.get("DEST_CITY_NAME", ""),
                "origin_country": row.get("ORIGIN_COUNTRY", ""),
                "dest_country": row.get("DEST_COUNTRY", ""),
            },
        )

    out = PROCESSED / "bts_t100_route_month_capacity_freight.csv"
    fields = [
        "origin",
        "dest",
        "year",
        "month",
        "origin_city",
        "dest_city",
        "origin_country",
        "dest_country",
        "departures_performed",
        "payload_lbs",
        "freight_lbs",
        "mail_lbs",
        "freight_mail_lbs",
        "freight_load_ratio",
        "avg_payload_per_departure_lbs",
        "avg_freight_mail_per_departure_lbs",
        "distance_miles",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, vals in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][0], kv[0][1])):
            origin, dest, year, month = key
            dep = vals["departures_performed"]
            payload = vals["payload_lbs"]
            freight_mail = vals["freight_lbs"] + vals["mail_lbs"]
            info = meta[key]
            writer.writerow(
                {
                    "origin": origin,
                    "dest": dest,
                    "year": year,
                    "month": month,
                    "origin_city": info["origin_city"],
                    "dest_city": info["dest_city"],
                    "origin_country": info["origin_country"],
                    "dest_country": info["dest_country"],
                    "departures_performed": round(dep, 4),
                    "payload_lbs": round(payload, 4),
                    "freight_lbs": round(vals["freight_lbs"], 4),
                    "mail_lbs": round(vals["mail_lbs"], 4),
                    "freight_mail_lbs": round(freight_mail, 4),
                    "freight_load_ratio": round(freight_mail / payload, 6) if payload > 0 else "",
                    "avg_payload_per_departure_lbs": round(payload / dep, 4) if dep > 0 else "",
                    "avg_freight_mail_per_departure_lbs": round(freight_mail / dep, 4) if dep > 0 else "",
                    "distance_miles": round(vals["distance_miles_weighted"] / max(dep, 1.0), 4),
                }
            )
    print(f"wrote {out} route-months={len(groups):,}")


def main() -> None:
    build_segment_sample()
    build_route_month_summary()


if __name__ == "__main__":
    main()
