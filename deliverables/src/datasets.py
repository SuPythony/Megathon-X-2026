"""Datasets used by GreenTwin: one verified list shared by SYSTEM_DESIGN.pdf and LLD.pdf.

Verified on 11 Oct 2026 against the publishers' pages unless the gotchas say "not re-verified".
"""

VERIFIED_DATE = "11 Oct 2026"

DATASETS = [
    dict(
        name="EPC domestic certificates + recommendations",
        publisher="MHCLG, Get energy performance data service",
        url="https://get-energy-performance-data.communities.gov.uk/",
        use="Part 0 homes · Part 3 score inputs · Part 2 rating check (certificate history) · Part 5 rules · "
            "recommendation labels",
        access="Bulk CSV download + developer API (OpenAPI 3, bearer token)",
        auth="GOV.UK One Login (register on day 1)",
        licence="OGL v3 except where stated (see the service's licensing restrictions page)",
        format="CSV; bulk = certificates lodged since 2012 (incl. expired or replaced)",
        coverage="England and Wales, per certificate",
        fields="CERTIFICATE_NUMBER (key; replaces the old LMK_KEY), POSTCODE, UPRN, UPRN_SOURCE, LOCAL_AUTHORITY, "
               "PROPERTY_TYPE, BUILT_FORM, CONSTRUCTION_AGE_BAND, TOTAL_FLOOR_AREA, FLOOR_HEIGHT, WALLS_/ROOF_/FLOOR_/"
               "WINDOWS_DESCRIPTION, MULTI_GLAZE_PROPORTION, GLAZED_TYPE, GLAZED_AREA, MAINHEAT_DESCRIPTION, MAIN_FUEL, "
               "MAINS_GAS_FLAG, MAINHEATCONT_DESCRIPTION, LOW_ENERGY_LIGHTING, PHOTO_SUPPLY, SOLAR_WATER_HEATING_FLAG, "
               "CURRENT_/POTENTIAL_ENERGY_RATING and _EFFICIENCY, ENERGY_CONSUMPTION_CURRENT, CO2_EMISSIONS_CURRENT, "
               "HEATING_/HOT_WATER_/LIGHTING_COST_CURRENT, TENURE, LODGEMENT_DATE, TRANSACTION_TYPE",
        gotchas="Recommendations link to certificates by CERTIFICATE_NUMBER (the API embeds them in each domestic EPC); "
                "the CSV data dictionary does not apply to API JSON names; POTENTIAL_* fields are leakage (labels only); "
                "keep the latest certificate per UPRN for the current state and the history for rating validation; "
                "addresses and UPRN are private join keys, dropped after joining.",
        how="# 1. Sign in with GOV.UK One Login, then: https://get-energy-performance-data.communities.gov.uk/data-access-options\n"
            "# 2. Bulk CSV download (filter to NW, SW, London local authorities) into data/raw/epc/\n"
            "# 3. Field definitions: .../download/data-dictionary?property_type=domestic\n"
            "# 4. API: bearer token via /login/authorize; OpenAPI spec at /api-technical-documentation",
    ),
    dict(
        name="EPC live tables (statistics)",
        publisher="MHCLG, Live tables on Energy Performance of Buildings Certificates",
        url="https://www.gov.uk/government/statistical-data-sets/live-tables-on-energy-performance-of-buildings-certificates",
        use="Part 0 calibration targets + download check · Part 1 twin validation · Part 3 peer context · dashboard",
        access="Direct ODS download, quarterly (data to June 2026, published 30 July 2026)",
        auth="None",
        licence="OGL v3",
        format="ODS, 0.3–16 MB per table; sheets for E&W, England, Wales, region, local authority",
        coverage="England and Wales; local authority × quarter from 2008/4",
        fields="D1/EB1: LA code, quarter, lodgements, total floor area, counts by band A–G; D3/EB3: floor area, "
               "habitable rooms, energy consumption (kWh/m²), CO2, fuel costs; EB4/NB4: lodgements by bungalow / flat & "
               "maisonette / house",
        gotchas="Counts of certificates lodged, not the dwelling stock (repeat certificates, opt-outs); 'all' tables mix "
                "new and existing dwellings; a Q1 2026 LA allocation error was revised in May 2026; recompute D3 averages "
                "from totals (December 2023 correction).",
        how="curl -L -o data/raw/live/D1.ods https://assets.publishing.service.gov.uk/media/6a69c502f23ff62c51c1a7df/D1-_Domestic_Properties.ods\n"
            "curl -L -o data/raw/live/D3.ods https://assets.publishing.service.gov.uk/media/6a69c5260d360fa159c37c91/D3-_Domestic_Properties.ods\n"
            "curl -L -o data/raw/live/EB4.ods https://assets.publishing.service.gov.uk/media/6a69c62b16bc92f51e1a42d9/EB4-_Existing_Domestic_Properties.ods\n"
            "# python: pd.read_excel('D1.ods', engine='odf', sheet_name='D1_by_LA', header=3)",
    ),
    dict(
        name="NEED anonymised data 2025",
        publisher="DESNZ, National Energy Efficiency Data-Framework",
        url="https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025",
        use="Part 2 EBM energy prior (training) · Part 0 synthetic behaviour calibration and validation",
        access="Direct download: 50,000-row and 4,000,000-row property-level samples + metadata",
        auth="None",
        licence="OGL v3",
        format="CSV/XLSX; 50k fits a spreadsheet, 4M for training",
        coverage="England and Wales, property level; gas and electricity 2005–2023",
        fields="annual gas and electricity consumption by year, property type, age band, floor-area band, region, IMD "
               "band, installed measures (loft, cavity, boiler) with years",
        gotchas="No postcode: join to EPC homes by archetype (type × age × floor area × region); values banded/rounded; "
                "a 2026 release may exist on the NEED collection page.",
        how="# Download the 4M-row sample + metadata from the gov.uk page into data/raw/need/",
    ),
    dict(
        name="Sub-national domestic electricity and gas by LSOA",
        publisher="DESNZ (accredited official statistics)",
        url="https://www.gov.uk/government/collections/sub-national-electricity-consumption-data",
        use="GNN area-model labels · dashboard observed layer · synthetic area check",
        access="Direct XLSX download (LSOA domestic electricity, LSOA domestic gas)",
        auth="None",
        licence="OGL v3",
        format="XLSX, 2010–2024 (2024 summary updated 19 Dec 2025)",
        coverage="England and Wales LSOAs (Scotland data zones), domestic meters only",
        fields="LSOA code, number of meters, total kWh, mean and median kWh per meter",
        gotchas="Domestic meters only; small areas suppressed (keep None, never 0); earlier years revised each release.",
        how="# Collection pages: sub-national electricity and gas consumption data -> LSOA domestic files into data/raw/desnz/",
    ),
    dict(
        name="UCL linked Price Paid + EPC, 1995–2024 (ReShare 857911)",
        publisher="UCL (Chi, Dennett, Oléron-Evans, Morphet), UK Data Service ReShare",
        url="https://reshare.ukdataservice.ac.uk/857911/",
        use="Part 4 Bayesian hedonic value model (training) · green twins",
        access="Open download, no registration: 857911_bundle_1.zip (3 GB) + bundle_2.zip (4 GB)",
        auth="None",
        licence="CC BY 4.0",
        format="Zipped CSV; >22M transactions, 106 variables (tranall_link_26122024)",
        coverage="England and Wales, sales 1 Jan 1995 – 31 Oct 2024",
        fields="price, date, priceper (price per m²), classt, lad23cd, LMK_KEY (EPC id), UPRN (post-Nov 2021), floor "
               "area, habitable rooms, EPC rating fields",
        gotchas="The linked file is the initial, uncleaned version: dedupe, drop non-market and outliers. ReShare 854942 "
                "(2011–19, 5.7M sales, cleaner) is a cross-check, but its linked CSV is CC BY-NC-SA (non-commercial).",
        how="wget -P data/raw/ucl https://reshare.ukdataservice.ac.uk/857911/   # follow the bundle links on the page",
    ),
    dict(
        name="HM Land Registry Price Paid Data",
        publisher="HM Land Registry",
        url="https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads",
        use="Part 4 recent sales after Oct 2024 (needs linking to EPC)",
        access="Direct CSV (complete file, yearly, monthly)",
        auth="None",
        licence="OGL v3 with attribution (address data has extra rights notes)",
        format="CSV; complete file ≈ 5 GB",
        coverage="England and Wales, 1995 → latest month",
        fields="transaction id, price, date, postcode, property type, new build, tenure, PPD category (A market / B other)",
        gotchas="No floor area or EPC (link by address/UPRN); drop category B. Standard URL, not re-verified this session.",
        how="# Download the yearly files for 2024–2026 from the page into data/raw/ppd/",
    ),
    dict(
        name="UK House Price Index",
        publisher="HM Land Registry / ONS",
        url="https://landregistry.data.gov.uk/app/ukhpi",
        use="Part 4 market adjustment (deflate prices to today)",
        access="CSV download",
        auth="None",
        licence="OGL v3",
        format="CSV",
        coverage="Local authority × property type × month, 1995 →",
        fields="index and average price by property type",
        gotchas="Use the LA × type index matching each sale. Standard URL, not re-verified this session.",
        how="# Download full UK HPI CSV from the app's download page into data/raw/hpi/",
    ),
    dict(
        name="Smart meter energy consumption, London households (Low Carbon London)",
        publisher="UK Power Networks via London Datastore",
        url="https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d",
        use="Part 1 electricity layer: half-hourly load-shape archetypes",
        access="Direct download (combined CSV ≈ 10 GB unzipped, ~167M rows; tariff file)",
        auth="None",
        licence="Open (see the London Datastore page)",
        format="CSV, half-hourly",
        coverage="5,567 London households, Nov 2011 – Feb 2014; ~1,100 on a dynamic time-of-use tariff in 2013",
        fields="household id, tariff group (Std / ToU), timestamp, kWh per half hour; ToU price schedule",
        gotchas="Electricity only, London only, 2011–14 (state as a limitation); stream with DuckDB; a smaller TU Delft "
                "refactored copy (CC BY) exists; UKDS SN 7857 is cleaner but restricted.",
        how="# Download the combined CSV from the data.gov.uk / London Datastore links; query in place with DuckDB",
    ),
    dict(
        name="Open-Meteo Historical Weather API (ERA5)",
        publisher="Open-Meteo (Copernicus ERA5 / ERA5-Land)",
        url="https://open-meteo.com/en/docs/historical-weather-api",
        use="Part 0/1 monthly heating degree days and irradiance per postcode district",
        access="REST GET, JSON; cache every response",
        auth="None for non-commercial use (commercial needs an API key)",
        licence="Data CC BY 4.0; cite Open-Meteo and ERA5 with the Copernicus acknowledgement",
        format="JSON",
        coverage="ERA5 0.25° from 1940, ERA5-Land 0.1° from 1950",
        fields="daily temperature_2m_mean, shortwave_radiation_sum (MJ/m²)",
        gotchas="A timezone parameter is required for daily variables; compute HDD (base 15.5 °C) from daily means.",
        how='curl "https://archive-api.open-meteo.com/v1/archive?latitude=53.45&longitude=-2.22&start_date=2024-01-01'
            '&end_date=2024-12-31&daily=temperature_2m_mean,shortwave_radiation_sum&timezone=Europe/London"',
    ),
    dict(
        name="Met Office HadUK-Grid",
        publisher="Met Office via CEDA",
        url="https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb",
        use="Documented degree days and 30–50-year regional trends (dashboard)",
        access="CEDA archive download",
        auth="CEDA registration",
        licence="OGL",
        format="NetCDF, 1 km / 5 km / regional",
        coverage="UK, daily and monthly temperature",
        fields="tas, tasmax, tasmin",
        gotchas="Large NetCDF files; use regional/5 km products for trends. Not re-verified this session.",
        how="# Register at CEDA, then download monthly tas for the needed years",
    ),
    dict(
        name="PVGIS 5.3 API",
        publisher="EU Joint Research Centre",
        url="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en",
        use="Part 1 PV yield (monthly PVcalc, hourly seriescalc)",
        access="REST GET at https://re.jrc.ec.europa.eu/api/v5_3/{PVcalc|seriescalc|tmy}",
        auth="None (30 calls/s per IP)",
        licence="Free use with attribution to EU JRC PVGIS",
        format="JSON / CSV",
        coverage="Europe incl. UK",
        fields="monthly and hourly PV output per kWp for azimuth and tilt",
        gotchas="Use the versioned v5_3 path; PVGIS aspect 0 = south; cache per postcode district.",
        how='curl "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=51.48&lon=-2.52&peakpower=1&loss=14&aspect=0'
            '&angle=35&outputformat=json"',
    ),
    dict(
        name="ONS Postcode Directory (ONSPD)",
        publisher="ONS Open Geography Portal",
        url="https://geoportal.statistics.gov.uk/",
        use="The join key: postcode → LSOA 2021, local authority, region, lat/lon",
        access="CSV download (May 2026 release; August 2026 user guide published)",
        auth="None",
        licence="OGL (contains OS and Royal Mail data; attribution required)",
        format="Zipped CSV ≈ 235 MB, quarterly",
        coverage="UK postcodes, live and terminated",
        fields="pcds, lsoa21cd, local authority code, region code, lat, long (names vary slightly by release)",
        gotchas="Use the latest release; keep only district + LSOA after joining (privacy).",
        how="# Open Geography Portal -> 'ONS Postcode Directory (May 2026)' -> download CSV into data/raw/onspd/",
    ),
    dict(
        name="English Indices of Deprivation 2025",
        publisher="MHCLG",
        url="https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025",
        use="Features for the EBM prior, value model and GNN; dashboard",
        access="Direct download (File 1 IMD; download-all tool at deprivation.communities.gov.uk)",
        auth="None",
        licence="OGL v3",
        format="XLSX/CSV",
        coverage="33,755 LSOAs (2021 geography), England; published 30 Oct 2025",
        fields="LSOA code, IMD score, rank, decile, domain scores",
        gotchas="England only; Wales uses WIMD (separate); use the corrected LSOA → LAD lookup (Aug 2025).",
        how="# deprivation.communities.gov.uk/download-all -> IMD 2025 LSOA file",
    ),
    dict(
        name="LSOA 2021 boundaries",
        publisher="ONS Open Geography Portal",
        url="https://geoportal.statistics.gov.uk/",
        use="Dashboard maps; LSOA adjacency graph for the GNN",
        access="GeoJSON / GeoPackage download",
        auth="None",
        licence="OGL",
        format="Generalised (BGC) for maps; full resolution for adjacency",
        coverage="England and Wales",
        fields="LSOA21CD, geometry",
        gotchas="Build adjacency (queen contiguity) once with geopandas/PySAL. Not re-verified this session.",
        how="# Open Geography Portal -> 'LSOA (December 2021) Boundaries EW BGC' -> GeoJSON",
    ),
    dict(
        name="Conservation areas + listed buildings",
        publisher="MHCLG planning.data.gov.uk (Historic England for listed buildings)",
        url="https://www.planning.data.gov.uk/dataset/conservation-area",
        use="Part 5 feasibility rules (EWI, glazing restrictions)",
        access="Direct CSV / GeoJSON / Parquet",
        auth="None",
        licence="OGL v3 (attribution: Historic England, Ordnance Survey)",
        format="10,666 conservation areas from 156 providers; 382,284 listed-building points",
        coverage="England (incomplete: not every authority has published)",
        fields="entity, name, geometry, organisation, start/end dates (listed buildings: list entry, grade, point)",
        gotchas="Incomplete coverage: unknown ≠ not constrained; flag the agent to confirm. Listed buildings are points "
                "inside the building, not its extent.",
        how="curl -L -o data/raw/planning/conservation-area.parquet https://files.planning.data.gov.uk/dataset/conservation-area.parquet\n"
            "curl -L -o data/raw/planning/listed-building.parquet https://files.planning.data.gov.uk/dataset/listed-building.parquet",
    ),
    dict(
        name="Off-gas-grid indicator",
        publisher="EPC MAINS_GAS_FLAG (primary); Xoserve / Centre for Sustainable Energy postcode lists (fallback)",
        url="https://data.london.gov.uk/dataset/postcodes-not-on-the-gas-network",
        use="Part 5 rules and grants (BUS off-gas uplift); dashboard layer",
        access="EPC field per home; postcode lists as XLSX/CSV",
        auth="None",
        licence="EPC: OGL; postcode lists: check terms (CSE 2017 listing has no licence)",
        format="Per home / postcode list",
        coverage="Postcode lists are dated (2013–2018)",
        fields="MAINS_GAS_FLAG (Y/N); postcode, OA/LSOA (CSE)",
        gotchas="Prefer the EPC flag; use postcode lists only where the EPC flag is missing, and say they are dated.",
        how="# Use MAINS_GAS_FLAG from the EPC bulk file",
    ),
    dict(
        name="Electrification of Heat heat-pump performance (SN 9050 / 9210 / 9209)",
        publisher="Energy Systems Catapult via UK Data Service",
        url="https://doc.ukdataservice.ac.uk/doc/9050/mrdoc/pdf/9050_eoh_Interim_insights_from_heat_pump_performance_data.pdf",
        use="Part 1 heat-pump SCOP priors by house type",
        access="SN 9050 cleansed data: open, summary dataset downloadable; full cleansed data by FTP order",
        auth="None for SN 9050 summary; SN 9049 raw data needs UKDS registration",
        licence="UK Data Service terms (check per study)",
        format="CSV (742 files in the full cleansed set); daily (SN 9210) and 30-minute (SN 9209) datasets",
        coverage="~750 heat pumps, Nov 2020 – Sep 2023 (2nd edition)",
        fields="seasonal performance factor, daily/half-hourly heat output and electricity input, house type",
        gotchas="Use the summary SPF by system type and house type as the prior; analysis code on GitHub "
                "(ES-Catapult/electrification_of_heat).",
        how="# UK Data Service catalogue: SN 9050 -> download summary dataset",
    ),
    dict(
        name="English Housing Survey energy tables",
        publisher="MHCLG",
        url="https://www.gov.uk/government/statistics/english-housing-survey-2022-to-2023-energy",
        use="Part 0 stock distributions to fill gaps in the synthetic population",
        access="Direct XLSX tables",
        auth="None",
        licence="OGL v3",
        format="XLSX",
        coverage="England dwelling stock",
        fields="wall type, insulation, glazing, heating by age and type",
        gotchas="Survey estimates (with sampling error); used for marginals only. Not re-verified this session.",
        how="# Download the energy report annex tables",
    ),
    dict(
        name="CubiCasa5k floor plans",
        publisher="CubiCasa / Aalto University (Zenodo record 2613548)",
        url="https://github.com/CubiCasa/CubiCasa5k",
        use="Part 1 photo-to-twin evaluation (room polygons, areas)",
        access="Zenodo download",
        auth="None",
        licence="Check the repository LICENSE; treat as non-commercial until confirmed",
        format="5,000 floor plans (4,200 / 400 / 400), SVG annotations per image",
        coverage="Finnish floor plans (style differs from UK plans)",
        fields="room polygons and types, walls, doors, windows",
        gotchas="Only for evaluating extraction quality; add a small set of our own labelled UK plans.",
        how="# Zenodo record 2613548 -> download archive",
    ),
    dict(
        name="Octopus Energy public products API (time-of-use tariff citation)",
        publisher="Octopus Energy",
        url="https://docs.octopus.energy/rest/guides/endpoints/",
        use="Part 2 time-of-use bill figures (cite one published tariff)",
        access="REST GET https://api.octopus.energy/v1/products/ ; product details list tariff unit-rate links",
        auth="None for product and tariff listings (consumption endpoints need a key)",
        licence="Octopus API terms; cite tariff, URL and access date in tariffs.json",
        format="JSON, paginated",
        coverage="Current and historical tariffs (e.g. Agile, Go)",
        fields="tariff codes, standard unit rates, standing charges, valid_from/valid_to",
        gotchas="Prices vary by region and day; record the exact tariff and date used. Until cited, ToU bills stay null.",
        how="curl https://api.octopus.energy/v1/products/",
    ),
]

DAY1 = [
    "Register GOV.UK One Login and get EPC bulk access (the slowest step); use cached samples meanwhile.",
    "Download the EPC live tables D1, D3, EB1, EB3, EB4 (ODS, minutes) and build area_epc_stats.",
    "Download NEED 2025 (50k now, 4M for training) with its metadata.",
    "Start the UCL 857911 bundles (≈ 7 GB) and the UK HPI CSV.",
    "Download ONSPD (latest), IMD 2025 and LSOA 2021 boundaries: the join keys for everything else.",
    "Download DESNZ LSOA domestic electricity and gas 2010–2024.",
    "Fetch Open-Meteo weather and PVGIS yields per postcode district (cache every response).",
    "Download conservation-area and listed-building files from planning.data.gov.uk.",
    "Later: London smart meters (10 GB, stream with DuckDB), Electrification of Heat SN 9050 summary, CubiCasa5k, "
    "one Octopus time-of-use tariff for tariffs.json.",
]

EPC_LIVE_USES = [
    ("Population calibration targets", "D1, EB4", "Rake our EPC sample and the synthetic population so band mix, "
     "property-type mix and floor area per local authority match the official totals for the same period."),
    ("Download / mapper check", "D1", "Our certificate counts per LA × quarter × band must reconcile with D1; a gap "
     "> 2% flags missing files, opt-outs or mapping errors."),
    ("Twin validation", "D3", "The twin's standard mode (SAP-like occupancy) aggregated per LA vs D3 mean modelled "
     "kWh/m² and fuel cost: an official check on the physics."),
    ("Score peer context", "D1, D3", "\"Your home vs your area\": local-authority band shares and median modelled "
     "kWh/m² in ScoreResult.peer_context."),
    ("Synthetic-data validation", "D1, D3, EB4", "Band, type and kWh/m² distributions of the synthetic homes vs the "
     "official tables per LA (chi-square, KS), reported in the evaluation."),
    ("Dashboard layer", "D1, D3", "Official LA layer beside our modelled LSOA layer: share A–C vs D–G, trend over "
     "quarters, mean modelled kWh/m², CO2 per dwelling."),
]
