import argparse
from logging import info
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import requests

from .commons import build_data_dir

GADM_URL = "https://biogeo.ucdavis.edu/data/gadm3.6/shp"
filename = lambda country_code: f"gadm36_{country_code}_shp.zip"

def download_shp_zip(location: Path, country_code: str) -> None:
    '''
    Downloads zipped shapefile from GADM to location.
    '''
    fname = filename(country_code)
    r = requests.get(f"{GADM_URL}/{fname}", stream = True)
    with (location/fname).open('wb') as tgt:
        for _ in r.iter_content(chunk_size = 512):
            tgt.write(_)

def get_GADM_data(data_root: str, country_codes: Dict[str, str], replace: bool = False) -> None:
    '''
    Downloads and unzips GADM files

    Inputs:
        - country_codes: mapping of country name to 3-letter code 
        - replace: (bool) if True will replace contents, if False will skip if country code has been processed already
    '''

    data_paths = build_data_dir(data_root)
    zip_dir = data_paths["root"]/"zipfiles"
    zip_dir.mkdir(exist_ok = True)

    for country_name, country_code in country_codes.items():
        outpath = data_paths['gadm']/country_code

        if replace or not outpath.is_dir():
            info("Downloading GADM file for %s", country_name)
            download_shp_zip(zip_dir, country_code)
            outpath.mkdir(exist_ok = True)
            with ZipFile(zip_dir/filename(country_code)) as z:
                z.extractall(outpath)

        else:
            print("GADM file for %s exists and replace set to False; skipping", country_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download GADM administrative boundaries globally')
    parser.add_argument("--replace", action='store_true', default=False)
    parser.add_argument("--data_root", type='str', require=True, description="Path to data folder")

    args = parser.parse_args()

    get_GADM_data(vars(args))
