import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple
import geopandas as gpd
import igraph
import numpy as np
import pandas as pd
import tqdm
from shapely.ops import polygonize
from shapely.wkt import dumps
from shapely.geometry import (LinearRing, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from logging import basicConfig, debug, info, warning, error

from .i_topology import PlanarGraph

def block_to_gadm(block: str) -> str:
    """Just grabs the GADM from a block"""
    block_rev = block[::-1]
    idx = block_rev.index("_") + 1
    return block[0:-idx]

def add_buildings(graph: PlanarGraph, 
                  bldg_centroids: List[Point],
                  ) -> PlanarGraph:
    """Add bldg centroids to planar graph"""

    total_blgds = len(bldg_centroids)
    for i, bldg_node in enumerate(bldg_centroids):
        graph.add_node_to_closest_edge(bldg_node, terminal=True)

    if total_blgds > 0:
        graph._cleanup_linestring_attr()
    return graph 

def clean_graph(graph: PlanarGraph) -> PlanarGraph:
    """
    Some graphs are malformed bc of bad geometries and 
    we take the largest connected subgraph
    """
    is_conn = graph.is_connected()
    if is_conn:
        info("Graph is connected...")
        return graph, 1
    else:
        info("Graph is NOT connected...")
        components = graph.components(mode=igraph.WEAK)
        num_components = len(components)
        comp_sizes = [len(idxs) for idxs in components]
        arg_max = np.argmax(comp_sizes)
        comp_indices = components[arg_max]

        debug("len comp_indices = {}".format(len(comp_indices)))
        debug("num_components = {}".format(num_components))
        debug("comp_sizes = {}".format(comp_sizes))
    
        sub = graph.subgraph(comp_indices)
        debug("Thru...")

        return graph.subgraph(comp_indices), num_components

def drop_buildings_intersecting_block(parcel_geom: LineString,
                                      bldg_centroids: List[Point],
                                      block_geom: Polygon, 
                                      ) -> List[Point]:
    """
    If a parcel shares a boundary with the block, then it already has access 
    and doesn't need to be included. So, polygonize the parcels and intersect 
    the parcel polygons with the boundary of the block, thus allowing reblocking
    to focus only on the interior parcels without access.

    Args:
        parcel_geom: Multilinestring of parcel boundaries in a block
        bldg_centroids: building centroid points
        block_geom: Polygon of block geometry
    
    Returns:
        Updated bldg_centroids list for reblocking targetting
    """
    # Converts the parcels to polygons
    parcel_geom_df = gpd.GeoDataFrame({'geometry': list(polygonize(parcel_geom))})
    parcel_geom_df = parcel_geom_df.explode()
    parcel_geom_df.reset_index(inplace=True, drop=True)

    # Make a dataframe of building points
    building_geom_df = gpd.GeoDataFrame({'geometry': [MultiPoint(bldg_centroids)]})
    building_geom_df = building_geom_df.explode()
    building_geom_df.reset_index(inplace=True, drop=True)
    building_geom_df.reset_index(inplace=True)
    building_geom_df.rename(columns={'index': 'building_id'}, inplace=True)

    # Map building to a parcel
    m = gpd.sjoin(parcel_geom_df, building_geom_df, how='left') 
    has_building = m['building_id'].notna()

    if has_building.sum() != building_geom_df.shape[0]:
        warning("Check drop_buildings_intersecting_block sjoin for block")
        warning("buildings = {} but matched = {}".format(building_geom_df.shape[0], has_building.sum()))
    m_has_building = m.loc[has_building]
    m_has_building = m_has_building.rename(columns={'geometry':'parcel_geom'})
    m_has_building = m_has_building.merge(building_geom_df, how='left', on='building_id')

    # Now check which parcel geoms intersect with the block
    block_boundary = block_geom.boundary 

    fn = lambda geom: geom.intersects(block_boundary)

    # And now return just the buildings that DO NOT have parcels on the border
    m_has_building['parcel_intersects_block'] = m_has_building['parcel_geom'].apply(fn)

    reblock_buildings = m_has_building[~m_has_building['parcel_intersects_block']]['geometry'].apply(lambda g: g.coords[0])
    return list(reblock_buildings.values)

def add_outside_node(block_geom: Polygon, 
                     bldg_centroids: List[Point],
                     ) -> List[Point]:
    """
    Steiner Tree Alg needs a point on the outside of the block so that
    it connects to the broader street network. This adds a dummy node.

    Args:
        block_geom: Polygon of block geometry
        bldg_centroids: building centroid points

    Returns:
        updated bldg_centroids w/ outside point added
    """
    bounding_rect = block_geom.minimum_rotated_rectangle
    convex_hull = block_geom.convex_hull
    outside_block = bounding_rect.difference(convex_hull)
    outside_building_point = outside_block.representative_point()
    bldg_centroids.append(outside_building_point.coords[0])
    
    return bldg_centroids 

def reblock(parcels: MultiLineString,
            buildings: MultiPolygon,
            block: Polygon,
            use_width: bool=False,
            simplify: bool=False,
            ) -> Tuple[MultiLineString, MultiLineString]:
    # [1] Bldg poly -> bldg centroids
    bldg_centroids = [b.centroid for b in buildings]

    # [2] Bldgs on block border have access already -- drop them
    bldg_centroids = drop_buildings_intersecting_block(parcels, 
                                                       bldg_centroids,
                                                       block,
                                                       )

    # [3] Add dummy bldg centroid to outside of block, forcing Steiner Alg
    #     to connect to the outside road network
    bldg_centroids = add_outside_node(block, bldg_centroids)

    # [4] Convert parcel -> PlanarGraph
    planar_graph = PlanarGraph.from_multilinestring(parcels)

    # [5] Add bldg centroids to PlanarGraph as terminal nodes
    add_buildings(planar_graph, bldg_centroids)

    # [6] Update the edge types to denote existing streets, per block geometry
    planar_graph.update_edge_types(block, check=True)

    # [7 Optional] Add width to PlanarGraph based on bldg geoms
    if use_width:
        planar_graph.set_edge_width(buildings, simplify=True)
        planar_graph.calc_edge_weight()

    # [8] Clean graph if it's disconnected
    planar_graph, num_components = clean_graph(planar_graph)

    # [9 Optional] Simplify the vertex structure
    if simplify:
        planar_graph.simplify()

    # [10] Steiner Approximation
    planar_graph.steiner_tree_approx()

    # [11] Extract optimal paths from graph and return
    new_path, existing_path = planar_graph.get_steiner_linestrings(expand=simplify)

    return planar_graph, new_path, existing_path
