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

# def update_edge_types(parcel_graph: PlanarGraph, 
#                       block_polygon: Polygon, 
#                       check: bool=False, 
#                       lines_pgraph: PlanarGraph=None,
#                       ) -> Tuple[int, int]:
#     """
#     When reblocking, existing roads receive 0 weight. Existing roads
#     are captured in the block_polygon, so given a graph and a 
#     block_geometry, this updates the edge types of the graph as
#     either existing / new. Then it updates the weights.
#     Graph is updated in place.

#     Args:
#         parcel_graph: graph repr of parcel boundaries
#         block_geom: Polygon of block geometry
#         check: If true, verify that each point in the block is in fact in the parcel
#         lines_pgraph: NA - will be deprecated

#     Returns:
#         Updates graph in place. Returns summary of matching
#         check between parcel and block
#     """
#     block_coords_list = list(block_polygon.exterior.coords)
#     coords = set(block_coords_list)

#     rv = (None, None)
#     missing = None
#     total = None
    
#     # Option to verify that each point in the block is in fact in the parcel
#     if check:
#         parcel_coords = set(v['name'] for v in parcel_graph.vs)
#         total = 0
#         is_in = 0
#         for coord in coords:
#             is_in = is_in+1 if coord in parcel_coords else is_in 
#             total += 1
#         missing = total-is_in
#         if missing != 0:
#             warning("{} of {} block coords are NOT in the parcel coords".format(missing, total)) 

#     # Get list of coord_tuples from the polygon
#     assert block_coords_list[0] == block_coords_list[-1], "Not a complete linear ring for polygon"

#     # Loop over the block coords (as define an edge) and update the corresponding edge type in the graph accordingly
#     # NOTE: every block coord will be within the parcel graph vertices
#     for i, n0 in enumerate(block_coords_list):
#         if i==0:
#             continue
#         else:
#             n1 = block_coords_list[i-1]
#             u_list = parcel_graph.vs.select(name_eq=n0)
#             v_list = parcel_graph.vs.select(name_eq=n1)
#             if len(u_list) > 0 and len(v_list) > 0:
#                 u = u_list[0]
#                 v = v_list[0]
#                 path_idxs = parcel_graph.get_shortest_paths(u, v, weights='weight', output='epath')[0]

#                 # the coords u and v from the block are
#                 if lines_pgraph is None:
#                     parcel_graph.es[path_idxs]['edge_type'] = 'highway'
#                 else:
#                     pass 
#                     # ft_type = get_feature_type_from_lines(lines_pgraph, n0, n1 )
#                     # parcel_graph.es[path_idxs]['edge_type'] = 'new'

#                     # # Now view our new graph
#                     # parcel_df = convert_to_gpd(parcel_graph)
#                     # print("\n\n changing to {}".format(ft_type))
#                     # plot_types(parcel_df)
#                     # plt.show()
#                     # parcel_graph.es[path_idxs]['edge_type'] = ft_type

#     parcel_graph.es.select(edge_type_eq='highway')['weight'] = 0

#     # WATERWAY_WEIGHT = NATURAL_WEIGHT = 1e4  # Not currently active
#     # parcel_graph.es.select(edge_type_eq='waterway')['weight'] = WATERWAY_WEIGHT
#     # parcel_graph.es.select(edge_type_eq='natural')['weight'] = NATURAL_WEIGHT

#     rv = (missing, total)
#     return rv 

