import argparse
import os
import sys
import time
import typing
from itertools import chain, combinations, permutations
from pathlib import Path
from typing import Dict, Tuple, Union
from logging import warning

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)
from shapely.ops import cascaded_union
from shapely.wkt import loads

from ..etl.commons import build_data_dir
from .i_topology import PlanarGraph


def point_to_node(point: Point) -> Tuple[float, float]:
    '''
    Helper function to convert shapely.Point -> Tuple
    '''
    return point.coords[0]

def load_complexity(complexity_path: str) -> gpd.GeoDataFrame:
    """Load complexity and convert to geodataframe"""
    complexity = pd.read_csv(complexity_path)    
    complexity = gpd.GeoDataFrame(complexity)
    complexity['geometry'] = complexity['geometry'].apply(loads)
    complexity.set_geometry('geometry', inplace=True)
    
    load_fn = lambda x: [point_to_node(p) for p in loads(x)]
    complexity['centroids_multipoint'] = complexity['centroids_multipoint'].apply(load_fn)
    return complexity

def load_reblock_inputs(data_paths: Dict[str, Path], 
                        region: str, 
                        gadm_code: str, 
                        gadm: str
                        ) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """
    Given a dict specifying paths, and a specified area defined by the
    region, gadm_code, and gadm, loads reblock inputs

    Args:
        data_paths: dict mapping strings to directories ex. {'blocks': path/to/blocks/, 'buildings': path/to/buildings/}
        region: geographic region
        gadm_code: 3-letter abbreviation for country
        gadm: full gadm label

    Returns:
        All inputs required for reblocking, including parcels, 
        building centroids, and block polygons.
    """

    complexity_path = os.path.join(data_paths['complexity'], region, gadm_code, "complexity_{}.csv".format(gadm))
    parcels_path = os.path.join(data_paths['parcels'], region, gadm_code, "parcels_{}.geojson".format(gadm))

    # Load the complexity file
    complexity = load_complexity(complexity_path)
    complexity.rename(columns={'centroids_multipoint': 'buildings'}, inplace=True)
    complexity['building_count'] = complexity['buildings'].apply(lambda x: len(x))

    # Split it into two dataframes
    buildings_df = complexity[['block_id', 'buildings', 'building_count']]
    blocks_df = complexity[['block_id', 'geometry']]

    # Now load the parcels
    parcels_df = gpd.read_file(parcels_path)
    return parcels_df, buildings_df, blocks_df 
