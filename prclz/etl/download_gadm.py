import wget
import zipfile
import os
import shutil
import wget
import pandas as pd 
import sys 
import argparse 
from pathlib import Path 

from .setup_paths import build_data_dir, TRANS_TABLE

def download_gadm_zip(country_code:str) -> None:
    '''
    Just pulls down the country zip file of GADM boundaries
    '''

    p = Path("./zipfiles")
    p.mkdir(exist_ok=True)
    url = "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_{}_shp.zip".format(country_code)
    wget.download(url, "./zipfiles")


def process_zip(country_code: str, outpath: str) -> None:
    '''
    Just unpacks the GADM country zip file and stores content

    Inputs:
        - country_code: (str) 3-letter code to identify country
        - outpath: (str) where to save the country's gadm data
    '''

    p = Path("./zipfiles")
    p = p / "gadm36_{}_shp.zip".format(country_code)

    with zipfile.ZipFile(p) as z:
        z.extractall(outpath)

def update_gadm_data(data_root: str, 
	                 replace: bool = False) -> None:
    '''
    Downloads all the GADM zip files, then unpacks the files

    Inputs:
        - replace: (bool) if True will replace contents, if False will skip if 
                          country code has been processed already

    '''

    data_paths = build_data_dir(data_root)

    global TRANS_TABLE
    df = TRANS_TABLE
    b = ~ df['gadm_name'].isna()
    codes = df[ b ]['gadm_name'].values
    names = df[ b ]['global_south_country'].values

    for country_name, country_code in zip(names, codes):
        print("\n\nProcessing GADM: ", country_name)
        outpath = data_paths['gadm'] / country_code

        if replace or not outpath.is_dir():
            print("\tdownloading...")
            download_gadm_zip(country_code)
            process_zip(country_code, outpath)

        else:
            print("\tskip, file present")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download GADM administrative boundaries globally')
    parser.add_argument("--replace", action='store_true', default=False)
    parser.add_argument("--data_root", type='str', require=True, description="Path to data folder")

    args = parser.parse_args()

    update_gadm_data(vars(args))
