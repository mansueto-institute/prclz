import os 
import geopandas as gpd 
import pandas as pd 
import numpy as np
import igraph
from pathlib import Path 
from typing import Callable, List, Dict 
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point, LineString

import i_topology
from i_topology import PlanarGraph
from i_topology_utils import csv_to_geo, update_edge_types
from i_reblock import add_buildings, clean_graph
from i_reblock import add_outside_node, drop_buildings_intersecting_block
from path_cost import FlexCost
import simplify_reblock

# Should just import the setup_paths.py script (eventually)
from ..data_processing.setup_paths import *

# ROOT = Path("/home/cooper/Documents/chicago_urban/mnp/cooper_prclz")

# # Below this, paths will be automatically set based on ROOT
# DATA_PATH = ROOT / "data"
# GEOFABRIK_PATH = DATA_PATH / "input"
# GEOJSON_PATH = DATA_PATH / "geojson"   

# BLOCK_PATH = DATA_PATH / "blocks"
# BLDGS_PATH = DATA_PATH / "buildings"
# PARCELS_PATH = DATA_PATH / "parcels"
# LINES_PATH = DATA_PATH / "lines"
# COMPLEXITY_PATH = DATA_PATH / "complexity"

def add_buildings_poly(parcel_poly_df: gpd.GeoDataFrame, 
                    building_list: List[Point],
                    planar_graph: i_topology.PlanarGraph,
                  ):

    def cost_fn(centroid: Point, edge: igraph.Edge) -> float:
        dist = LineString(planar_graph.edge_to_coords(edge)).distance(centroid)
        return dist 
        # width = edge['width']
        # if width == 0:
        #     return dist 
        # door_dist = 0.0002 # this is a hyperparameter
        # if dist > door_dist:
        #     return 10 + dist 
        # else:
        #     return 1 / width #  
        #return dist 

    if isinstance(building_list[0], tuple):
        temp = [Point(x) for x in building_list]
        building_list = temp 

    bldg_gdf = gpd.GeoDataFrame.from_dict({'geometry': building_list})
    bldg_joined = gpd.sjoin(bldg_gdf, parcel_poly_df, how='left')

    for _, (centroid, parcel_id) in bldg_joined[['geometry', 'parcel_id']].iterrows():
        edge_seq = planar_graph.edges_in_parcel(parcel_id)
        #print("Point {} parcel_id {} finds {} edges".format(centroid.wkt, parcel_id, len(edge_seq)))
        edge_cost = map(lambda e: cost_fn(centroid, e), edge_seq)
        argmin = np.argmin(list(edge_cost))

        closest_edge = edge_seq[argmin]
        planar_graph.add_bldg_centroid(centroid, closest_edge)


def load_reblock_inputs(region: str, gadm_code: str, gadm: str):

    # Paths
    parcels_path = os.path.join(PARCELS_PATH, region, gadm_code, "parcels_{}.geojson".format(gadm))
    buildings_path = os.path.join(BLDGS_PATH, region, gadm_code, "buildings_{}.geojson".format(gadm))
    blocks_path = os.path.join(BLOCK_PATH, region, gadm_code, "blocks_{}.csv".format(gadm))

    # Load the files
    parcels_df = gpd.read_file(parcels_path)
    buildings_df = gpd.read_file(buildings_path)
    blocks_df = csv_to_geo(blocks_path)
    blocks_df.rename(columns={'block_geom': 'geometry'}, inplace=True)
    blocks_df = blocks_df[['geometry', 'block_id']]
    blocks_df = gpd.GeoDataFrame(blocks_df, geometry='geometry')

    # Map buildings to a block
    # Convert buildings to centroids
    buildings_df['buildings'] = buildings_df['geometry'].centroid
    buildings_df.set_geometry('buildings', inplace=True)

    # We want to map each building to a given block to then map the buildings to a parcel
    buildings_df = gpd.sjoin(buildings_df[['buildings', 'osm_id', 'geometry']], blocks_df, how='left', op='within')
    buildings_df = buildings_df[['buildings', 'block_id', 'geometry']].groupby('block_id').agg(list)
    buildings_df['building_count'] = buildings_df['buildings'].apply(lambda x: len(x))
    buildings_df.reset_index(inplace=True)

    return parcels_df, buildings_df, blocks_df 

def reblock_gadm2(
    region: str, 
    gadm_code: str, 
    gadm: str, 
    cost_fn: Callable, 
    return_metric_closures: bool = True,
    return_planar_graphs: bool = True, 
    through_street_cutoff: float = 0.0,
    simplify_new_roads: bool = False,
    block_list: List[str] = None,
    ):

    # Load inputs
    parcels, buildings, blocks = load_reblock_inputs(region, gadm_code, gadm)

    if block_list is None:
        block_list = blocks['block_id'].values

    reblock_data = {'geometry': [], 'line_type': [], 'block_id': []}
    reblock_poly_data = {'geometry': [], 'line_type': [], 'block_id': []}
    metric_closures = {}
    planar_graphs = {}

    for block_id in block_list:
        print("Reblocking {}".format(block_id))
        reblock_output = reblock_block_id(parcels, buildings, blocks,
                                          block_id, cost_fn, 
                                          reblock_data=reblock_data, 
                                          reblock_poly_data=reblock_poly_data,
                                          return_metric_closure=return_metric_closures,
                                          simplify_new_roads=simplify_new_roads,
                                          through_street_cutoff=through_street_cutoff)
        if reblock_output is None:
            continue
        reblock_data, reblock_poly_data, planar_graph, metric_closure = reblock_output
        if return_metric_closures:
            metric_closures[block_id] = metric_closure
        if return_planar_graphs:
            planar_graphs[block_id] = planar_graph
    
    #reblock_data = gpd.GeoDataFrame.from_dict(reblock_data)
    #reblock_poly_data = gpd.GeoDataFrame.from_dict(reblock_poly_data)
    rv = [reblock_data, reblock_poly_data, parcels, buildings, blocks]

    if return_metric_closures: 
        rv.append(metric_closures)
    if return_planar_graphs:
        rv.append(planar_graphs)
    return rv 

def reblock_block_id(parcels: gpd.GeoDataFrame, 
                     buildings: pd.DataFrame,
                     blocks: gpd.GeoDataFrame,
                     block_id: str,
                     cost_fn: Callable,
                     return_metric_closure: bool = True,
                     simplify_new_roads: bool = False,
                     through_street_cutoff: float = 0.0,
                     reblock_data: Dict = None,
                     reblock_poly_data: Dict = None):

    if reblock_data is None:
        reblock_data = {'geometry': [], 'line_type': [], 'block_id': []}
    if reblock_poly_data is None:
        reblock_poly_data = {'geometry': [], 'line_type': [], 'block_id': []}

    # (0) Get data for the block
    parcel_geom = parcels[parcels['block_id']==block_id]['geometry'].iloc[0]
    bldg_df = buildings[buildings['block_id']==block_id]['buildings']
    if bldg_df.shape[0] == 0:
        return None 
    building_list = bldg_df.iloc[0]
    block_geom = blocks[blocks['block_id']==block_id]['geometry'].iloc[0]

    # (1) drop buildings that intersect with the block border -- they have access
    if len(building_list) <= 1:
        return None 
    building_list = drop_buildings_intersecting_block(parcel_geom, building_list, block_geom, block_id)
    if len(building_list) <= 1:
        return None 

    # (2) And explicitly add a dummy building outside of the block which will force Steiner Alg
    #      to connect to the outside road network
    dummy_bldg = []
    dummy_bldg = add_outside_node(block_geom, dummy_bldg)
    #building_list = add_outside_node(block_geom, building_list)

    # (3) Convert parcel geometry to planar graph
    if True:
        parcel_poly_list = list(polygonize(parcel_geom))
        planar_graph = PlanarGraph.multipolygon_to_planar_graph(parcel_poly_list)
        test_es = planar_graph.es.select(parcel_id_eq=None)
        assert len(test_es) == 0, "When creating graph there are some edges not mapped to a parcel"
        parcel_poly_df = gpd.GeoDataFrame.from_dict({'geometry':parcel_poly_list, 
                                                     'parcel_id':range(len(parcel_poly_list))
                                                    })
    else:
        planar_graph = PlanarGraph.multilinestring_to_planar_graph(parcel_geom)

    # Because bldg centroid assignment needs width, move that up
    # but simplify AFTER we add teh building centroids
    if True:
        bldg_polys = buildings[buildings['block_id']==block_id]['geometry'].iloc[0]
        planar_graph.set_edge_width(bldg_polys, simplify=False)

    # (4) Add building centroids to the planar graph
    #bldg_tuples = [list(b.coords)[0] for b in building_list]
    #planar_graph = add_buildings(planar_graph, bldg_tuples)
    #planar_graph = add_buildings(planar_graph, building_list)
    add_buildings_poly(parcel_poly_df, building_list, planar_graph)
    planar_graph = add_buildings(planar_graph, dummy_bldg)
    planar_graph.set_edge_width(bldg_polys, simplify=True)

    # (5) Clean the graph if its disconnected
    planar_graph, num_components = clean_graph(planar_graph)

    # (6) Update the planar graph if the cost_fn needs it
    if cost_fn.lambda_turn_angle > 0:
        planar_graph.set_node_angles()

    # (7) The current existing roads should have 0 weight
    missing, total_block_coords = update_edge_types(planar_graph, block_geom, check=True)

    # (8) Do steiner approximation
    metric_closure = planar_graph.flex_steiner_tree_approx(cost_fn = cost_fn, return_metric_closure=return_metric_closure)
    
    # (9) Add through-streets, if enabled
    if through_street_cutoff > 0:
        through_lines = simplify_reblock.get_through_lines(planar_graph,
                                                           metric_closure,
                                                           through_street_cutoff,
                                                           cost_fn,
                                                           )        
    # (10) Get the linestrings 
    new_lines, existing_lines, new_polys, existing_polys = planar_graph.get_steiner_linestrings(expand=False, 
                                                                        return_polys=True)
    

    # (11) Simplify new road geometries
    if simplify_new_roads:
        new_lines = simplify_reblock.simplify_reblocked_graph(planar_graph)
        new_lines = unary_union(new_lines)

    reblock_data['geometry'] += [new_lines, existing_lines]
    reblock_data['line_type'] += ['new', 'existing']
    reblock_data['block_id'] += [block_id, block_id ]
    
    reblock_poly_data['geometry'] += [new_polys, existing_polys]
    reblock_poly_data['line_type'] += ['new', 'existing']
    reblock_poly_data['block_id'] += [block_id, block_id ]

    return reblock_data, reblock_poly_data, planar_graph, metric_closure 


def reblock_gadm(region: str, 
                 gadm_code: str, 
                 gadm: str, 
                 block_list: List[str], 
                 only_block_list: List[str] = False, 
                 drop_already_completed: bool = True, 
                 mins_threshold: float = np.inf,
                 ):
    
    # Make cost_fn
    lambda_width = 1.0
    through_street_cutoff = 0.7
    simplify_new_roads = True 

    cost_fn = FlexCost(lambda_width = lambda_width)
    reblock_output = reblock2.reblock_gadm2(region, gadm_code, gadm, cost_fn, 
                                            block_list=block_list,
                                            return_metric_closures=True,
                                            return_planar_graphs=True,
                                            through_street_cutoff=through_street_cutoff,
                                            simplify_new_roads=simplify_new_roads)
    return reblock_output 


# def reblock_gadm2(
#     region: str, 
#     gadm_code: str, 
#     gadm: str, 
#     cost_fn: Callable, 
#     return_metric_closures: bool = True,
#     return_planar_graphs: bool = True, 
#     through_street_cutoff: float = 0.0,
#     simplify_new_roads: bool = False,
#     block_list: List[str] = None,
#     ):

# # #cost_fn = reblock2.FlexCost(lambda_width=1.0,lambda_degree=200., lambda_turn_angle=2.)
# cost_fn = reblock2.FlexCost()
# parcels, buildings, blocks = reblock2.load_reblock_inputs(region, gadm_code, gadm)
# planar_graph = PlanarGraph.multilinestring_to_planar_graph(parcel_geom)

# bldg_tuples = [list(b.coords)[0] for b in building_list]
# planar_graph = reblock2.add_buildings(planar_graph, bldg_tuples)

# planar_graph.set_node_angles()
# bldg_polys = buildings[buildings['block_id']==block_id]['geometry'].iloc[0]
# planar_graph.set_edge_width(bldg_polys)

# planar_graph, num_components = reblock2.clean_graph(planar_graph)


