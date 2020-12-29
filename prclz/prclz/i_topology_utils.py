import typing 
from typing import Union, Tuple 
from itertools import combinations, chain, permutations

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point, LineString
from shapely.ops import cascaded_union
from shapely.wkt import loads
import pandas as pd
import numpy as np 
import time 

import os 
import matplotlib.pyplot as plt 
import sys 
from typing import Dict 

import argparse
from .i_topology import PlanarGraph

from ..data_processing.setup_paths import build_data_dir, TRANS_TABLE


def point_to_node(point: Point) -> Tuple[float]:
    '''
    Helper function to convert shapely.Point -> Tuple
    '''
    return point.coords[0]

def csv_to_geo(csv_path, add_file_col=False) -> gpd.GeoDataFrame:
    '''
    Given a path to a block.csv file, returns as a GeoDataFrame
    '''

    df = pd.read_csv(csv_path, usecols=['block_id', 'geometry'])

    # Block id should unique identify each block
    assert df['block_id'].is_unique, "Loading {} but block_id is not unique".format(csv_path)

    df.rename(columns={"geometry":"block_geom"}, inplace=True)
    df['block_geom'] = df['block_geom'].apply(loads)

    if add_file_col:
        f = csv_path.split("/")[-1]
        df['gadm_code'] = f.replace("blocks_", "").replace(".csv", "")

    return gpd.GeoDataFrame(df, geometry='block_geom')

def load_geopandas_files(region: str, gadm_code: str, 
                         gadm: str) -> (gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame):

    bldgs_path = os.path.join(BLDGS_PATH, region, gadm_code, "buildings_{}.geojson".format(gadm))
    lines_path = os.path.join(LINES_PATH, region, gadm_code, "lines_{}.geojson".format(gadm))
    parcels_path = os.path.join(PARCELS_PATH, region, gadm_code, "parcels_{}.geojson".format(gadm))
    blocks_path = os.path.join(BLOCK_PATH, region, gadm_code, "blocks_{}.csv".format(gadm))

    bldgs = gpd.read_file(bldgs_path)
    blocks = csv_to_geo(blocks_path)
    parcels = gpd.read_file(parcels_path)

    return bldgs, blocks, parcels, None

def load_reblock_inputs(data_paths: Dict, region: str, gadm_code: str, gadm: str):

    complexity_path = os.path.join(data_paths['complexity'], region, gadm_code, "complexity_{}.csv".format(gadm))
    parcels_path = os.path.join(data_paths['parcels'], region, gadm_code, "parcels_{}.geojson".format(gadm))

    # Load the complexity file
    complexity = pd.read_csv(complexity_path)    
    complexity = gpd.GeoDataFrame(complexity)
    complexity['geometry'] = complexity['geometry'].apply(loads)
    complexity.rename(columns={'centroids_multipoint': 'buildings'}, inplace=True)
    complexity.set_geometry('geometry', inplace=True)
    load_fn = lambda x: [point_to_node(p) for p in loads(x)]
    complexity['buildings'] = complexity['buildings'].apply(load_fn)
    complexity['building_count'] = complexity['buildings'].apply(lambda x: len(x))

    # Split it into two dataframes
    buildings_df = complexity[['block_id', 'buildings', 'building_count']]
    blocks_df = complexity[['block_id', 'geometry']]

    # Now load the parcels
    parcels_df = gpd.read_file(parcels_path)
    return parcels_df, buildings_df, blocks_df 

def prepare_parcels(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, 
                                               parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    '''
    For a single GADM, this script (1) creates the PlanarGraph associated
    with each respective parcel and (2) maps all buildings to their corresponding
    parcel. The buildings are converted to centroids and then to Node types so
    they can just be added to the PlanarGraph
    '''

    # Convert buildings to centroids
    bldgs['centroids'] = bldgs['geometry'].centroid
    bldgs.set_geometry('centroids', inplace=True)

    # We want to map each building to a given block to then map the buildings to a parcel
    bldgs = gpd.sjoin(bldgs, blocks, how='left', op='within')
    bldgs.drop(columns=['index_right'], inplace=True)

    # Now, join the parcels with the buildings
    parcels = parcels.merge(bldgs[['block_id', 'centroids']], how='left', on='block_id')
    parcels.rename(columns={'geometry':'parcel_geometry', 'centroids':'buildings'}, inplace=True)

    # Now collapse on the block and clean
    parcels = parcels.groupby('block_id').agg(list)
    parcels['parcel_geometry'] = parcels['parcel_geometry'].apply(lambda x: x[0])
    parcels['buildings'] = parcels['buildings'].apply(lambda x: [] if x==[np.nan] else x)

    # Checks
    assert blocks.shape[0] == parcels.shape[0]  # We should maintain block count
    parcels['buildings_count'] = parcels['buildings'].apply(lambda x: len(x))
    #assert parcels['buildings_count'].sum() == bldgs.shape[0]  # We should maintain bldgs count

    parcels.reset_index(inplace=True)

    # Now, create the graph for each parcel
    parcels['planar_graph'] = parcels['parcel_geometry'].apply(PlanarGraph.multilinestring_to_planar_graph)

    # And convert the buildings from shapely.Points -> topology.Nodes
    parcels['buildings'] = parcels['buildings'].apply(lambda x: [point_to_node(p) for p in x])

    return parcels 


def edge_list_from_linestrings(lines_df):
    '''
    Extract the geometry from 
    '''
    all_edges = []
    lines_df_geom = lines_df.geometry 
    for l in lines_df_geom:
        l_graph = PlanarGraph.linestring_to_planar_graph(l, False)
        l_graph_edges = l_graph.es 
        all_edges.extend(l_graph_edges)
    return all_edges


def check_block_parcel_consistent(block: MultiPolygon, parcel: MultiLineString):

    block_coords = block.exterior.coords 
    parcel_coords = list(chain.from_iterable(l.coords for l in parcel))

    for block_coord in block_coord:
        assert block_coord in parcel_coords


def update_edge_types(parcel_graph: PlanarGraph, block_polygon: Polygon, check=False, lines_pgraph=None):

    block_coords_list = list(block_polygon.exterior.coords)
    coords = set(block_coords_list)

    rv = (None, None)
    missing = None
    total = None
    # Option to verify that each point in the block is in fact in the parcel
    if check:
        parcel_coords = set(v['name'] for v in parcel_graph.vs)
        total = 0
        is_in = 0
        for coord in coords:
            is_in = is_in+1 if coord in parcel_coords else is_in 
            total += 1
        missing = total-is_in
        #print("{} of {} block coords are NOT in the parcel coords".format(missing, total)) 

    # Get list of coord_tuples from the polygon
    assert block_coords_list[0] == block_coords_list[-1], "Not a complete linear ring for polygon"

    # Loop over the block coords (as define an edge) and update the corresponding edge type in the graph accordingly
    # NOTE: every block coord will be within the parcel graph vertices
    for i, n0 in enumerate(block_coords_list):
        if i==0:
            continue
        else:
            n1 = block_coords_list[i-1]
            u_list = parcel_graph.vs.select(name_eq=n0)
            v_list = parcel_graph.vs.select(name_eq=n1)
            if len(u_list) > 0 and len(v_list) > 0:
                u = u_list[0]
                v = v_list[0]
                path_idxs = parcel_graph.get_shortest_paths(u, v, weights='weight', output='epath')[0]

                # the coords u and v from the block are
                if lines_pgraph is None:
                    parcel_graph.es[path_idxs]['edge_type'] = 'highway'
                else:
                    ft_type = get_feature_type_from_lines(lines_pgraph, n0, n1 )
                    parcel_graph.es[path_idxs]['edge_type'] = 'new'

                    # Now view our new graph
                    parcel_df = convert_to_gpd(parcel_graph)
                    print("\n\n changing to {}".format(ft_type))
                    plot_types(parcel_df)
                    plt.show()
                    parcel_graph.es[path_idxs]['edge_type'] = ft_type

    parcel_graph.es.select(edge_type_eq='highway')['weight'] = 0
    #print("There are {} edges with type highway".format(len(parcel_graph.es.select(edge_type_eq='highway'))))

    WATERWAY_WEIGHT = NATURAL_WEIGHT = 1e4  # Not currently active
    parcel_graph.es.select(edge_type_eq='waterway')['weight'] = WATERWAY_WEIGHT
    parcel_graph.es.select(edge_type_eq='natural')['weight'] = NATURAL_WEIGHT

    rv = (missing, total)
    return rv 

###########################################################################################
###########################################################################################
# NOTE: this section is all code used to recover the feature type (i.e. waterway, road, natural)
#       contained within OSM but not contained within the block files

def convert_to_gpd(g):

    if 'edge_type' not in g.es.attributes():
        g.es['edge_type'] = None 
        
    edge_geom = [LineString(g.edge_to_coords(e)) for e in g.es]
    edge_types = g.es['edge_type']

    df = pd.DataFrame(data={'geometry':edge_geom, 'edge_type': edge_types})
    return gpd.GeoDataFrame(df)

def plot_types(g):

    edge_color_map = {'new': 'red', None: 'orange', 'waterway': 'blue', 
                      'highway': 'black', 'natural': 'green', 'gadm_boundary': 'orange'}
    ax = g[g['edge_type'].isna()].plot(color='red')

    for t in g.edge_type.unique():
        d = g[g['edge_type'] == t]
        if d.shape[0] > 0:
            d.plot(ax=ax, color=edge_color_map[t])


def create_lines_graph(lines: gpd.GeoDataFrame) -> PlanarGraph:
    '''
    Create a PlanarGraph based on a lines GeoDataFrame. The graph will
    have a feature_type attribute for the edges
    '''

    b_waterway = ((lines['highway']=="") & (lines['natural']=="")) | (lines['waterway'].notna())
    b_highway = ((lines['waterway']=="") & (lines['natural']=="")) | (lines['highway'].notna())
    b_natural = ((lines['highway']=="") & (lines['waterway']=="")) | (lines['natural'].notna())

    lines['feature_type'] = None 
    lines.loc[b_waterway,'feature_type']='waterway' 
    lines.loc[b_highway,'feature_type']='highway' 
    lines.loc[b_natural,'feature_type']='natural' 
    assert np.all(lines['feature_type'].notna())

    pgraph = PlanarGraph()
    for index, row in lines[['feature_type','geometry']].iterrows():
        ft = row['feature_type']
        coords_list = list(row['geometry'].coords)
        for i, coords in enumerate(coords_list):
            if i == 0:
                continue 
            else:
                pgraph.add_edge(coords, coords_list[i-1], feature_type=ft)
    return pgraph 

def get_feature_type_from_lines(lines_pgraph: PlanarGraph, coords0, coords1 ) -> str:

    # edge0_ft = lines_pgraph.add_node_to_closest_edge(coords0, get_edge=True)['feature_type']
    # edge1_ft = lines_pgraph.add_node_to_closest_edge(coords1, get_edge=True)['feature_type']
    edge0, dist0 = lines_pgraph.add_node_to_closest_edge(coords0, get_edge=True)
    edge1, dist1 = lines_pgraph.add_node_to_closest_edge(coords1, get_edge=True)

    # If our 'closest edge' is in fact relatively far from any line, then the block point
    #    is actually probably from the GADM boundary
    edge0_ft = 'gadm_boundary' if dist0 > THRESHOLD_METERS else edge0['feature_type']
    edge1_ft = 'gadm_boundary' if dist1 > THRESHOLD_METERS else edge1['feature_type']

    if edge0_ft != edge1_ft:
        print("block coords are different types --> coord0 = {} | coord1 = {}".format(edge0_ft, edge1_ft))

        if 'highway' in (edge0_ft, edge1_ft):
            return 'highway'
        else:
            return 'natural'
    else:
        return edge0_ft


###########################################################################################
###########################################################################################

if __name__ == "__main__":

    BLOCKS = '../data/blocks/Africa/KEN'
    LINES = BLOCKS.replace("DJI", "KEN").replace("blocks", "lines")

    gadm = 'KEN.30.10.1_1'
    lines = gpd.read_file(os.path.join(LINES, 'lines_{}.geojson'.format(gadm)))
    blocks = csv_to_geo(os.path.join(BLOCKS, 'blocks_{}.csv'.format(gadm)))

    b_waterway = ((lines['highway']=="") & (lines['natural']=="")) | (lines['waterway'].notna())
    b_highway = ((lines['waterway']=="") & (lines['natural']=="")) | (lines['highway'].notna())
    b_natural = ((lines['highway']=="") & (lines['waterway']=="")) | (lines['natural'].notna())

    lines['feature_type'] = None 
    lines.loc[b_waterway,'feature_type']='waterway' 
    lines.loc[b_highway,'feature_type']='highway' 
    lines.loc[b_natural,'feature_type']='natural' 
    assert np.all(lines_df['feature_type'].notna())

    fig, ax = plt.subplots(1,2)
    blocks.plot(alpha=0.2, color='black', ax=ax[0])

    colors = {'waterway':'blue', 'highway':'red', 'natural':'green'}

    for f in ['waterway', 'highway', 'natural']:

        d = lines[lines['feature_type']==f]
        if d.shape[0] > 0:
            d.plot(color=colors[f], ax=ax[1])

