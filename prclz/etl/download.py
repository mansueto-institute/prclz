from logging import error, info
from pathlib import Path
from typing import Dict, Optional, Sequence
from zipfile import ZipFile

import pandas as pd
import requests
from urlpath import URL

GADM_URL      = URL("https://biogeo.ucdavis.edu/data/gadm3.6/shp")
GEOFABRIK_URL = URL("http://download.geofabrik.de/")

def build_data_dir(root: str, additional: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    '''
    Build canonical data directory structure
    '''
    root = Path(root)
    data_paths = {
        folder: root/folder for folder in 
        ['blocks', 'buildings', 'cache', 'complexity', 'errors', 'gadm', 'geofabrik', 'geojson', 'geojson_gadm', 'input', 'lines', 'parcels'] + 
        (additional if additional else [])
    }
    data_paths["root"] = root       

    for v in data_paths.values():
        v.mkdir(parents=True, exist_ok=True)

    return data_paths

def download(src: URL, dst: Path) -> None:
    r = requests.get(src, stream = True)
    with dst.open('wb') as fd:
        for content in r.iter_content(chunk_size = 512):
            fd.write(content)

def gadm_filename(country_code) -> str: 
    return f"gadm36_{country_code}_shp.zip"

def geofabrik_filename(region, name) -> str:
    if name == "antarctica":
        return "antarctica-latest.osm.pbf"
    elif name == "puerto-rico":
        return "north-america/us/puerto-rico-latest.osm.pbf"
    return f"{region}/{name}-latest.osm.pbf"

def get_gadm_data(data_root: str, country_codes: Dict[str, str], overwrite: bool = False) -> None:
    '''
    Downloads and unzips GADM files

    Inputs:
        - country_codes: mapping of country name to 3-letter code 
        - overwrite: (bool) if True will overwrite contents, if False will skip if country code has been processed already
    '''

    data_paths = build_data_dir(data_root, additional = ["zipfiles"])

    for (country_name, country_code) in country_codes.items():
        outpath = data_paths['gadm']/country_code

        if overwrite or not outpath.is_dir():
            info("Downloading GADM file for %s", country_name)
            filepath = gadm_filename(country_code)
            try: 
                download(GADM_URL/filepath, data_paths["zipfiles"]/filepath) 
                with ZipFile(data_paths["zipfiles"]/filepath) as z:
                    z.extractall(outpath)
            except Exception as e:
                error("Error downloading and extracting shapefile for %s: %s", country_name, e)

        else:
            info("GADM file for %s exists and overwrite set to False; skipping", country_name)

def get_geofabrik_data(data_root: str, country_regions: Dict[str, str], overwrite: bool = False) -> None:
    '''
    Given a geofabrik country name and the corresponding region, downloads the 
    geofabrik pbf file which contains all OSM data for that country. Checks whether
    the data has already been downloaded
    '''
    data_paths = build_data_dir(data_root)
    for (name, region) in country_regions.items():
        outpath = data_paths["geofabrik"]/f"{name}-latest.osm.pbf"
        if overwrite or not outpath.exists():
            info("Downloading Geofabrik file for %s/%s", region, name)
            try: 
                download(GEOFABRIK_URL/geofabrik_filename(region, name), outpath)
            except Exception as e:
                error("Error downloading PBF for %s/%s: %s", region, name, e)
        else:
            info("Geofabrik file for %s/%s exists and overwrite set to False; skipping", region, name)


def main(data_source: str, data_root: str, country_codes: Optional[Sequence[str]], overwrite: bool):
    mappings = pd.read_csv(Path(__file__).parent/"country_codes.csv")
    if country_codes:
        mappings = mappings[mappings.gadm_name.isin(country_codes)]
    if data_source.lower() == "gadm":
        gadm_mapping = mappings.dropna()\
            [["country", "gadm_name"]]\
            .set_index("country")\
            .to_dict()["gadm_name"]
        get_gadm_data(data_root, gadm_mapping, overwrite)
    if data_source.lower() == "geofabrik":
        geofabrik_mapping = mappings.dropna()\
            [["geofabrik_name", "geofabrik_region"]]\
            .set_index("geofabrik_name")\
            .to_dict()["geofabrik_region"]
        get_geofabrik_data(data_root, geofabrik_mapping, overwrite)
    # argument parser in cli.py validates that data source will be valid