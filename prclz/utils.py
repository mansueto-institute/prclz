import os
from pathlib import Path
from logging import info, warning
from typing import Tuple, Union

import geopandas as gpd
import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon

from .topology import Node, PlanarGraph


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

def load_geopandas_files(region: str, gadm_code: str, gadm: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

    bldgs_path = os.path.join(BLDGS_PATH, region, gadm_code, "buildings_{}.geojson".format(gadm))
    lines_path = os.path.join(LINES_PATH, region, gadm_code, "lines_{}.geojson".format(gadm))
    parcels_path = os.path.join(PARCELS_PATH, region, gadm_code, "parcels_{}.geojson".format(gadm))
    blocks_path = os.path.join(BLOCK_PATH, region, gadm_code, "blocks_{}.csv".format(gadm))

    bldgs = gpd.read_file(bldgs_path)
    blocks = csv_to_geo(blocks_path)
    parcels = gpd.read_file(parcels_path)
    lines = gpd.read_file(lines_path)

    return bldgs, blocks, parcels, lines

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
    parcels['planar_graph'] = parcels['parcel_geometry'].apply(PlanarGraph.from_multilinestring)

    # And convert the buildings from shapely.Points -> topology.Nodes
    parcels['buildings'] = parcels['buildings'].apply(lambda x: [Node.from_point(p) for p in x])

    return parcels 


def edge_list_from_linestrings(lines_df):
    '''
    Extract the geometry from 
    '''
    all_edges = []
    lines_df_geom = lines_df.geometry 
    for l in lines_df_geom:
        l_graph = PlanarGraph.from_linestring(l, False)
        #l_graph_edges = [Edge(e) for e in l_graph.edges]
        l_graph_edges = [e for e in l_graph.edges]
        all_edges.extend(l_graph_edges)
    return all_edges


def update_graph_with_edge_type(graph, lines:gpd.GeoDataFrame):
    '''
    Split the lines DataFrame into lists of edges of type 'waterway', 'highway', 
    and 'natural'. Then loop over the graph's edges and update the weights 
    and the edge_type accordingly
    '''

    waterway_edges = edge_list_from_linestrings(lines[lines['waterway'].notna()])
    natural_edges = edge_list_from_linestrings(lines[lines['natural'].notna()])
    highway_edges = edge_list_from_linestrings(lines[lines['highway'].notna()])

    for u, v, edge_data in graph.edges(data=True):
        edge_tuple = (u,v)
        if edge_tuple in waterway_edges:
            edge_data['weight'] = 999
            edge_data['edge_type'] = "waterway"     
        elif edge_tuple in natural_edges:
            edge_data['weight'] = 999
            edge_data['edge_type'] = "natural" 
        elif edge_tuple in highway_edges:
            edge_data['weight'] = 0
            edge_data['edge_type'] = "highway" 
        else:
            edge_data['edge_type'] = "parcel"
