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

def to_gdf(shape):
    if isinstance(shape, list):
        return gpd.GeoDataFrame.from_dict({'geometry':shape})
    else:
        return gpd.GeoDataFrame.from_dict({'geometry':[shape]})

def viz_reblock(parcel_geom: MultiLineString,
                bldg_geoms: MultiPolygon,
                block_geom: Polygon,
                new_steiner: MultiLineString,
                exist_steiner: MultiLineString,
                ):
    parcel_geom = to_gdf(parcel_geom)
    bldg_geoms = to_gdf(bldg_geoms)
    block_geom = to_gdf(block_geom)
    new_steiner = to_gdf(new_steiner)
    exist_steiner = to_gdf(exist_steiner)
    
    ax = block_geom.plot(edgecolor='black', alpha=0.1)
    ax = parcel_geom.plot(edgecolor='black', alpha=0.5, ax=ax)
    ax = bldg_geoms.plot(color='black', ax=ax)
    ax = new_steiner.plot(color='red', ax=ax)
    ax = exist_steiner.plot(color='green', ax=ax)
    return ax

def debug_load():
    base = Path('/home/cooper/Documents/chicago_urban/mnp/prclz-proto/data')
    blocks_p = base / "blocks" / "Africa" / "DJI" / "blocks_DJI.1.1_1.csv"
    bldgs_p = base / "buildings" / "Africa" / "DJI" / "buildings_DJI.1.1_1.geojson"
    parcels_p = base / "parcels" / "Africa" / "DJI" / "parcels_DJI.1.1_1.geojson"

    blocks = pd.read_csv(blocks_p)    
    blocks = gpd.GeoDataFrame(blocks)
    blocks['geometry'] = blocks['geometry'].apply(loads)

    bldgs = gpd.read_file(bldgs_p)
    parcels = gpd.read_file(parcels_p)

    return parcels, bldgs, blocks

# Use 'DJI.1.1_1_1'
def extract(block_id: str,
            parcels: gpd.GeoDataFrame,
            bldgs: gpd.GeoDataFrame,
            blocks: gpd.GeoDataFrame,
            ) -> Tuple[MultiLineString, MultiPolygon, Polygon]:
    block = blocks[blocks['block_id']==block_id].iloc[0]['geometry']
    parcel = parcels[parcels['block_id']==block_id].iloc[0]['geometry']
    bldg_sel = list(bldgs[bldgs.intersects(block)]['geometry'].values)
    return parcel, bldg_sel, block

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
