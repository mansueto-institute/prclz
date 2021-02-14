import argparse
from logging import basicConfig, error, info
from typing import Dict
from zipfile import ZipFile

from urlpath import URL

from .commons import build_data_dir, download

GADM_URL = URL("https://biogeo.ucdavis.edu/data/gadm3.6/shp")
filename = lambda country_code: f"gadm36_{country_code}_shp.zip"

def load_country_codes(csv_path: str) -> Dict[str, str]:
    pass 

def fetch_data(data_root: str, country_codes: Dict[str, str], replace: bool = False) -> None:
    '''
    Downloads and unzips GADM files

    Inputs:
        - country_codes: mapping of country name to 3-letter code 
        - replace: (bool) if True will replace contents, if False will skip if country code has been processed already
    '''

    data_paths = build_data_dir(data_root, additional = ["zipfiles"])

    for (country_name, country_code) in country_codes.items():
        outpath = data_paths['gadm']/country_code

        if replace or not outpath.is_dir():
            info("Downloading GADM file for %s", country_name)
            filepath = filename(country_code)
            try: 
                download(GADM_URL/filepath, data_paths["zipfiles"]/filepath) 
                with ZipFile(data_paths["zipfiles"]/filepath) as z:
                    z.extractall(outpath)
            except Exception as e:
                error("Error downloading shapefile for %s: %s", country_name, e)

        else:
            info("GADM file for %s exists and replace set to False; skipping", country_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download GADM administrative boundaries globally')
    parser.add_argument("--data_root",     type='str', require=True, description="Path to data folder")
    parser.add_argument("--country_codes", type='str', require=True, description="Path to country codes CSV")
    parser.add_argument("--replace",       action='store_true', default=False)

    args = parser.parse_args()
    basicConfig(level = "INFO")
    fetch_data(vars(args))
