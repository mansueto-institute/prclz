from typing import List, Union
from logging import basicConfig, info
from pathlib import Path 
from ..utils import gadm_dir_to_path

import geopandas as gpd
import pandas as pd

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


def main(
    bldg_file: str,
    gadm_path: str,
    output_dir: str,
    ) -> None:
    """
    Given a path to a geojson containing all the building polygons
    and a path to the GADM boundary file, splits the master building
    file by GADM and saves a file for each GADM in the output dir

    Args:
        bldg_file: path to geojson w/ building polygons
        gadm_path: path to GADM w/ gadm region polygons. NOTE, for convenience
                   also accepts directory containing all gadm files and finds the
                   highest res file within directory
        output_dir: directory to save the resutling gadm-specific bldg files

    Returns:
        None, saves out files to output_dir
    """
    gadm_path = Path(gadm_path)
    if gadm_path.is_dir():
        gadm_file = gadm_dir_to_path(gadm_path)
        info("Loading GADM file: %s", gadm_file)
    else:
        gadm_file = gadm_path
    bldgs = gpd.read_file(str(bldg_file))
    gadms = gpd.read_file(str(gadm_file))
    gadms = clean_gadm_cols(gadms)

    bldgs = gpd.sjoin(bldgs, gadms, how='left', op='intersects')
    bldgs.drop(columns=['index_right'], inplace=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for gadm, data in bldgs.groupby('gadm'):
        f = "buildings_{}.geojson".format(gadm)
        out_path = output_dir / f

        data.to_file(out_path, driver='GeoJSON')
        info("Creating file: %s", str(out_path))

    return bldgs, gadms
