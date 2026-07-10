# Disease Data Cleaning Notes

Run the reproducible cleaner from the project root:

```bash
python3 disease_data/clean_datasets.py
```

The cleaner preserves raw files and regenerates the analysis-ready CSVs in
`data/interim` and `data/processed`.

## Cleaned Files

- `data/raw/geojson-counties-fips.json`: source county GeoJSON used to filter California counties for choropleth mapping.
- `data/interim/copd_ca_county_clean.csv`: one row per California county. Columns are `state`, `county`, `county_fips`, and `copd_adjusted_prevalence_pct`.
- `data/interim/retail_esd_availability_clean.csv`: one row per California county for retail ESD availability in 2016. Columns are `state`, `county`, `county_fips`, `year`, `retail_esd_availability_prop`, and `retail_esd_availability_pct`.
- `data/processed/ca_county_copd_retail_esd_merged.csv`: county-level merge for correlation analysis. Use `retail_esd_availability_pct` and `copd_adjusted_prevalence_pct` when comparing percent-scale values.

## Interactive Map Outputs

- `results/figures/copd-adjusted-prevalence-choropleth.html`: California county choropleth for 2023 age-adjusted adult COPD prevalence.
- `results/figures/retail-esd-availability-choropleth.html`: California county choropleth for 2016 retail electronic smoking device availability.
- `results/figures/ca-counties.geojson`: California county GeoJSON used by the choropleth maps.
- `results/figures/map-stats.json`: summary statistics used for the mini project write-up.

## Cleaning Decisions

- County names are trimmed and matched to five-digit `county_fips`.
- The retail source includes Berkeley, Long Beach, Pasadena, and STATEWIDE rows. These are not county observations, so they are excluded.
- Retail availability is sourced from the 2016 rows to keep one county-level observation per county.
- Retail source values are decimals from 0 to 1. The cleaned file keeps that value as `retail_esd_availability_prop` and adds `retail_esd_availability_pct` on a 0 to 100 scale.
- No retail values are imputed for the 2016 county-level analysis set.
