from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_RETAIL_FILE = BASE_DIR / "data" / "raw" / "retail-availability-of-electronic-smoking-devices-by-county.csv"
COPD_INTERIM_FILE = BASE_DIR / "data" / "interim" / "copd_ca_county_clean.csv"
RETAIL_INTERIM_FILE = BASE_DIR / "data" / "interim" / "retail_esd_availability_clean.csv"
MERGED_PROCESSED_FILE = BASE_DIR / "data" / "processed" / "ca_county_copd_retail_esd_merged.csv"

RETAIL_ANALYSIS_YEAR = 2016
EXPECTED_CA_COUNTY_COUNT = 58
NON_COUNTY_RETAIL_AREAS = {"Berkeley", "Long Beach", "Pasadena", "STATEWIDE"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def parse_decimal(value: str, field: str, context: str) -> Decimal:
    cleaned = clean_text(value)
    if cleaned in {"", "*"}:
        raise ValueError(f"Missing or suppressed {field} for {context}")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"Could not parse {field}={value!r} for {context}") from exc


def format_decimal(value: Decimal, places: int) -> str:
    quantizer = Decimal("1").scaleb(-places)
    return f"{value.quantize(quantizer):f}"


def clean_copd_rows() -> list[dict[str, str]]:
    rows = []
    seen_fips: set[str] = set()
    seen_counties: set[str] = set()

    for row in read_csv(COPD_INTERIM_FILE):
        state = clean_text(row["state"]).upper()
        county = clean_text(row["county"])
        county_fips = clean_text(row["county_fips"]).zfill(5)
        prevalence = parse_decimal(
            row["copd_adjusted_prevalence_pct"],
            "copd_adjusted_prevalence_pct",
            county,
        )

        if state != "CA":
            raise ValueError(f"Unexpected state in COPD data: {state!r}")
        if county_fips in seen_fips:
            raise ValueError(f"Duplicate COPD county_fips: {county_fips}")
        if county in seen_counties:
            raise ValueError(f"Duplicate COPD county: {county}")

        seen_fips.add(county_fips)
        seen_counties.add(county)
        rows.append(
            {
                "state": state,
                "county": county,
                "county_fips": county_fips,
                "copd_adjusted_prevalence_pct": format_decimal(prevalence, 1),
            }
        )

    rows.sort(key=lambda row: row["county_fips"])
    if len(rows) != EXPECTED_CA_COUNTY_COUNT:
        raise ValueError(f"Expected 58 COPD counties, found {len(rows)}")
    return rows


def clean_retail_rows(copd_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fips_by_county = {row["county"]: row["county_fips"] for row in copd_rows}
    retail_by_county: dict[str, dict[str, str]] = {}

    for row in read_csv(RAW_RETAIL_FILE):
        county = clean_text(row["County"])
        if county in NON_COUNTY_RETAIL_AREAS:
            continue

        year = int(clean_text(row["Year"]))
        if year != RETAIL_ANALYSIS_YEAR:
            continue

        if county not in fips_by_county:
            raise ValueError(f"Retail county not found in COPD/FIPS lookup: {county}")
        if county in retail_by_county:
            raise ValueError(f"Duplicate retail row for {county} in {RETAIL_ANALYSIS_YEAR}")

        availability_prop = parse_decimal(row["Percentage"], "Percentage", f"{county} {year}")
        if availability_prop < 0 or availability_prop > 1:
            raise ValueError(f"Retail availability proportion out of range for {county}: {availability_prop}")

        retail_by_county[county] = {
            "state": "CA",
            "county": county,
            "county_fips": fips_by_county[county],
            "year": str(year),
            "retail_esd_availability_prop": format_decimal(availability_prop, 3),
            "retail_esd_availability_pct": format_decimal(availability_prop * Decimal("100"), 1),
        }

    missing_counties = sorted(set(fips_by_county) - set(retail_by_county))
    if missing_counties:
        raise ValueError(f"Missing retail rows for counties: {missing_counties}")

    rows = sorted(retail_by_county.values(), key=lambda row: row["county_fips"])
    if len(rows) != EXPECTED_CA_COUNTY_COUNT:
        raise ValueError(f"Expected 58 retail counties, found {len(rows)}")
    return rows


def merge_rows(
    copd_rows: list[dict[str, str]],
    retail_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    copd_by_fips = {row["county_fips"]: row for row in copd_rows}
    merged = []

    for retail_row in retail_rows:
        copd_row = copd_by_fips.get(retail_row["county_fips"])
        if copd_row is None:
            raise ValueError(f"No COPD match for county_fips {retail_row['county_fips']}")

        merged.append(
            {
                "state": retail_row["state"],
                "county": retail_row["county"],
                "county_fips": retail_row["county_fips"],
                "retail_year": retail_row["year"],
                "retail_esd_availability_prop": retail_row["retail_esd_availability_prop"],
                "retail_esd_availability_pct": retail_row["retail_esd_availability_pct"],
                "copd_adjusted_prevalence_pct": copd_row["copd_adjusted_prevalence_pct"],
            }
        )

    return merged


def main() -> None:
    copd_rows = clean_copd_rows()
    retail_rows = clean_retail_rows(copd_rows)
    merged_rows = merge_rows(copd_rows, retail_rows)

    write_csv(
        COPD_INTERIM_FILE,
        copd_rows,
        ["state", "county", "county_fips", "copd_adjusted_prevalence_pct"],
    )
    write_csv(
        RETAIL_INTERIM_FILE,
        retail_rows,
        [
            "state",
            "county",
            "county_fips",
            "year",
            "retail_esd_availability_prop",
            "retail_esd_availability_pct",
        ],
    )
    write_csv(
        MERGED_PROCESSED_FILE,
        merged_rows,
        [
            "state",
            "county",
            "county_fips",
            "retail_year",
            "retail_esd_availability_prop",
            "retail_esd_availability_pct",
            "copd_adjusted_prevalence_pct",
        ],
    )

    print(f"Wrote {len(copd_rows)} rows to {COPD_INTERIM_FILE}")
    print(f"Wrote {len(retail_rows)} rows to {RETAIL_INTERIM_FILE}")
    print(f"Wrote {len(merged_rows)} rows to {MERGED_PROCESSED_FILE}")


if __name__ == "__main__":
    main()
