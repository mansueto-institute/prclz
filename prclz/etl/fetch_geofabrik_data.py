import pandas as pd
import time 
import typing
import requests

import os 
import sys 
import wget

from .setup_paths import build_data_dir, TRANS_TABLE


def urlexists_stream(uri: str) -> bool:
    '''
    Tests whether a URL is a valid address 
    '''
    try:
        with requests.get(uri, stream=True) as response:
            try:
                response.raise_for_status()
                return True
            except requests.exceptions.HTTPError:
                return False
    except requests.exceptions.ConnectionError:
        return False

def make_url(geo_name: str, geo_region: str) -> str:
    '''
    Simple helper to construct the geofabrik download URL given the name and region
    '''

    url = "http://download.geofabrik.de/{}/{}-latest.osm.pbf".format(geo_region, geo_name)

    if geo_name == "antarctica":
        url = "http://download.geofabrik.de/antarctica-latest.osm.pbf"

    elif geo_name == "puerto-rico":
        url = "http://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf"

    return url 

def download_data(geofabrik_name: str, 
                  geofabrik_region: str,
                  geofabrik_path: str,
                  ) -> None:
    '''
    Given a geofabrik country name and the corresponding region, downloads the 
    geofabrik pbf file which contains all OSM data for that country. Checks whether
    the data has already been downloaded
    '''

    outfile = geofabrik_name + "-latest.osm.pbf"


    # Check that we haven't already downloaded it
    output_path = os.path.join(geofabrik_path, geofabrik_region.title())
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if os.path.isfile(os.path.join(output_path, outfile)):
        print("\nWe have geofabrik data for {} -- see: {}\n".format(geofabrik_name, output_path))

    else:
        url = make_url(geofabrik_name, geofabrik_region)

        if urlexists_stream(url):
            wget.download(url, os.path.join(output_path, outfile))
            print("\nSuccesfully downloaded geofabrik data for {}".format(geofabrik_name))
        else:
            print("\ngeofabrik_name = {} or geofabrik_region = {} are wrong\n".format(geofabrik_name, geofabrik_region))

def update_geofabrik_data(data_root: str, 
                          replace: bool = False) -> None:
    
    data_paths = build_data_dir(data_root)
    geofabrik_path = data_paths['geofabrik']
    global TRANS_TABLE    

    names = TRANS_TABLE['geofabrik_name']
    regions = TRANS_TABLE['geofabrik_region']

    for geofabrik_name, geofabrik_region in zip(names, regions):

        download_data(geofabrik_name, geofabrik_region, geofabrik_path)
   

