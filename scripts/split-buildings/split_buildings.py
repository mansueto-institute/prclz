
import pygeos
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon
from typing import List, Union
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence
import zipfile
import requests
from urlpath import URL
import os
import shutil
import argparse
import time

#os.environ["USE_PYGEOS"] = "0/1"
gpd.options.use_pygeos = True
pd.options.mode.chained_assignment = None 

def gadm_dir_to_path(gadm_dir: Union[str, Path]) -> str:
    """
    For a given country,the GADM dir contains multiple file levels
    so this convenience function just returns the path to the highest
    resolution gadm file within the directory, which changes from 
    country to country
    Args:
        gadm_dir: directory containing all gadm files for a country
    
    Returns:
        Path to specific gadm file
    """
    sort_fn = lambda p: int(p.stem.split("_")[-1])
    gadm_dir = Path(gadm_dir)
    files = [p for p in gadm_dir.iterdir() if ".shp" in p.name]
    files.sort(key=sort_fn)
    return files[-1]


def clean_gadm_cols(gadms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    The GADM files have columns GID_0, GID_1, ... , GID_n depending
    on the gadm resolution. So this extracts the highest res gadms
    and standardizes the col names
    """
    sort_fn = lambda n: int(n.replace("GID_",""))
    cols = [c for c in gadms.columns if "GID_" in c]
    cols.sort(key=sort_fn)

    cc = cols[0]
    gadm = cols[-1]
    std_gadms = gadms[[cc, gadm, 'geometry']]
    std_gadms.rename(columns={cc: 'gadm_code', gadm: 'gadm'}, inplace=True)
    return std_gadms


def main(log_path: Path, codes_file: Path, progress_file: Path, gadm_path: Path, input_dir: Path, output_dir: Path):
    """
    Given a path to file containing concordance between building polygons
    and country codes, a path to the GADM boundary file, splits the master 
    building file by GADM and saves a file for each GADM in the output dir
    Args:
        log_path: directory to hold the .log file
        codes_file: file containing building file names and country codes 
                    (columns = ['file_name','country_code'])
        progress_file: file containing progress on the codes_file (either completed or invalid Zip files)
                    (columns = ['file_name','country_code','outcome']
        gadm_path: path to GADM w/ gadm region polygons. NOTE, for convenience
                   also accepts directory containing all gadm files and finds the
                   highest res file within directory
        input_dir: directory containing building files
        output_dir: directory to save the resutling gadm-specific bldg files
    Returns:
        None, saves out files to output_dir
    """
    logging.basicConfig(filename=Path(log_path), level=logging.INFO)
    logging.info('Started')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = pd.read_csv(codes_file)

    if os.path.isfile(progress_file):
        progress = pd.read_csv(progress_file)
        skip_list = list(progress[progress['outcome'] == 'completed']['file_name'])
        codes = codes[~codes['file_name'].isin(skip_list)]

    for code, files in codes.groupby('country_code'): 

        gadm_folder = Path(gadm_path) / code
        if gadm_folder.is_dir():
            gadm_file = gadm_dir_to_path(gadm_folder)
            logging.info("Loading GADM file: %s", gadm_file)
        else:
            gadm_file = gadm_folder
    
        codes_sub = codes[codes['country_code'] == code]
        output_folder = Path(output_dir) / code
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for index, row in codes_sub.iterrows():
            t1 = time.time()
            file_path = Path(input_dir) / row[0]
            try:
                z = zipfile.ZipFile(file_path)
            except: 
                logging.info("Invalid Zip file : %s", str(row[0]))
                entry = pd.DataFrame({'file_name': [row[0]], 'country_code': [code], 'outcome':['invalid']}) 
                if not os.path.isfile(progress_file):
                    entry.to_csv(progress_file, header='column_names', index=False)
                else: 
                    file_log = pd.read_csv(progress_file)
                    file_log = file_log.append(entry)
                    file_log.to_csv(progress_file, mode='w', index=False)
                continue
            logging.info("Extracting : %s", str(input_dir))
            z.extractall(path=input_dir) 
            file_name = [y for y in sorted(z.namelist()) if y.endswith('shp')] 
            logging.info("Reading : %s", str(file_name))
            bldgs = gpd.read_file(Path(input_dir) / file_name[0].replace("//", "/"))
            bldgs = bldgs.to_crs(epsg=4326)
            folder_name = os.path.dirname(Path(input_dir) / file_name[0].replace("//", "/"))
            logging.info("Deleting : %s", str(folder_name))
            shutil.rmtree(folder_name) 
            t2 = time.time()
            logging.info("Building read time: %s", str(t2-t1) )
            
            gadms = clean_gadm_cols(gpd.read_file(gadm_file))
            t3 = time.time()
            bldgs['osm_id'] = range(len(bldgs))
            logging.info("Spatial join: %s", str(gadm_file))
            bldgs_index = bldgs.sindex
            index_bulk = bldgs_index.query_bulk(gadms['geometry'], predicate="intersects")
            gadm_bldgs_map = pd.DataFrame({'index_gadm': index_bulk[0], 'index_bldgs': index_bulk[1]})
            bldgs_mapped = gadm_bldgs_map.merge(bldgs, left_on='index_bldgs', right_index=True)
            bldgs_mapped = bldgs_mapped.merge(gadms[['gadm','gadm_code']], left_on='index_gadm', right_index=True)
            bldgs_mapped = gpd.GeoDataFrame(bldgs_mapped[['osm_id','gadm','gadm_code','geometry']])
            t4 = time.time()
            logging.info("Query bulk: %s", str(t4-t3) )

            for gadm, data in bldgs_mapped.groupby('gadm'):
                out_path = Path(output_dir) / code / f"buildings_{gadm}.geojson"
                if out_path.exists():
                    logging.info("Combining file: %s", str(out_path))
                    bldgs_append = gpd.read_file(out_path)
                    data = bldgs_append.append(data)
                data.to_file(out_path, driver='GeoJSON')
                logging.info("Creating file: %s", str(out_path))
            t5 = time.time()
            logging.info("Completed: %s", str(row[0] + str(t5-t4)))
            entry = pd.DataFrame({'file_name': [row[0]], 'country_code': [code], 'outcome':['completed']})
            if not os.path.isfile(progress_file):
                entry.to_csv(progress_file, header='column_names', index=False)
            else: 
                file_log = pd.read_csv(progress_file)
                file_log = file_log.append(entry)
                file_log.to_csv(progress_file, mode='w', index=False)

    logging.info('Finished')


def setup(args=None):
    logging.getLogger().setLevel("DEBUG")
    parser = argparse.ArgumentParser(description='Run building splitter.')
    parser.add_argument('--log_path', required=True, type=Path, dest="log_path")
    parser.add_argument('--codes_file', required=True, type=Path, dest="codes_file")
    parser.add_argument('--progress_file', required=True, type=Path, dest="progress_file")
    parser.add_argument('--gadm_path', required=True, type=Path, dest="gadm_path")
    parser.add_argument('--input_dir', required=True, type=Path, dest="input_dir")
    parser.add_argument('--output_dir', required=True, type=Path, dest="output_dir")

    return parser.parse_args(args)

if __name__ == "__main__":
    main(**vars(setup()))




