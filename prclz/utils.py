from logging import info, warning
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon


def parse_ona_text(text: str) -> Polygon:
    str_coordinates = text.split(';')
    coordinates = [s.split() for s in str_coordinates]
    return Polygon([(float(x), float(y)) for (y, x, t, z) in coordinates])

def get_gadm_level_column(gadm: gpd.GeoDataFrame, level: int = 5) -> str:
    gadm_level_column = "GID_{}".format(level)
    while gadm_level_column not in gadm.columns and level > 0:
        warning("GID column for GADM level %s not found, trying with level %s", level, level-1)
        level -= 1
        gadm_level_column = "GID_{}".format(level)
    info("Using GID column for GADM level %s", level)
    return gadm_level_column, level

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

def csv_to_geo(csv_path, add_file_col=False):
    '''
    Loads the csv and returns a GeoDataFrame
    '''

    df = pd.read_csv(csv_path, usecols=['block_id', 'geometry'])

    # Block id should unique identify each block
    assert df['block_id'].is_unique, "Loading {} but block_id is not unique".format(csv_path)

    df.rename(columns={"geometry":"block_geom"}, inplace=True)
    df['block_geom'] = df['block_geom'].apply(shapely.wkt.loads)

    if add_file_col:
        f = csv_path.split("/")[-1]
        df['gadm_code'] = f.replace("blocks_", "").replace(".csv", "")

    return gpd.GeoDataFrame(df, geometry='block_geom', crs='EPSG:4326')