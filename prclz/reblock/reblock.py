from logging import warning, info
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import geopandas as gpd
import igraph
import pandas as pd
import tqdm
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon)
from shapely.ops import nearest_points, polygonize

from ..utils import csv_to_geo
from .reblock_graph import ReblockGraph


def block_to_gadm(block: str) -> str:
    """Just grabs the GADM from a block"""
    return block[0:-(block[::-1].index("_") + 1)]

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
        warning("Buildings = %s but matched = %s", building_geom_df.shape[0], has_building.sum())
    m_has_building = m.loc[has_building]
    m_has_building = m_has_building.rename(columns={'geometry':'parcel_geom'})
    m_has_building = m_has_building.merge(building_geom_df, how='left', on='building_id')

    # Now check which parcel geoms intersect with the block
    block_boundary = block_geom.boundary 

    # And now return just the buildings that DO NOT have parcels on the border
    m_has_building['parcel_intersects_block'] = m_has_building['parcel_geom'].apply(lambda geom: geom.intersects(block_boundary))

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

def snap_block(
    block: Polygon,
    parcels: gpd.GeoSeries,
    ) -> LineString:
    """
    We need to identify which points in the parcels are from
    the original block, which signifies existing streets. There
    is a small eps difference in the points resulting from the
    parcelization process so this maps each block point to 
    its nearest vertex in the parcel. Note, we compare vertices
    because downstream this will be used to identify points
    in an iGraph graph, rather than a shapely geometry.

    Args:
        block: polygon representing existing streets
        parcels: parcel boundaries

    Returns:
        A linestring repr of the block, where vertices 
        have been mapped to those in parcels
    """
    if not isinstance(block, Polygon):
        warning("block is type: %s", type(block))
    block_bounds = block.boundary
    parcel_bounds = parcels.boundary.unary_union

    parcel_bounds_pts = []
    for g in parcel_bounds:
        parcel_bounds_pts.extend(list(g.coords))
    parcel_bounds_pts = MultiPoint(parcel_bounds_pts)

    if isinstance(block_bounds, LineString):
        mapped_pts = [nearest_points(Point(b), parcel_bounds_pts)[1] for b in block_bounds.coords] 
    else:
        warning("block_bounds is type: %s", type(block_bounds))
        mapped_pts = []
        for block_bound in block_bounds:
            mapped_pts.extend([nearest_points(Point(b), parcel_bounds_pts)[1] for b in block_bound.coords])

    return LineString(mapped_pts)


def reblock(parcels: Union[MultiLineString, MultiPolygon],
            buildings: MultiPolygon,
            block: Polygon,
            use_width: bool = False,
            simplify_roads: bool = False,
            thru_streets_top_n: int = 0,
            ) -> Tuple[ReblockGraph, MultiLineString, MultiLineString]:
    """
    Perform reblocking given parcels, target buildings, and the 
    block (i.e. the existing streets). Performs Steiner Tree approximation
    and uses the Euclidean distance as the weight by default. 
    Includes option to use width when calculating weighst and 
    for adding a post-processing stage where the roads are simplified.

    Args:
        parcels: parcel boundaries capturing possible optimal roads
        buildings: the buildings which will be our reblocking target
        block: existing road network
        use_width: if True, calculates weights as w = distance/width
        thru_streets_top_n: if not None, will add thru streets to
                            mitigated dead ends resulting from reblocking.
                            Adds the top-k most severe dead ends
        simplify_roads: if True, does post-processing to make roads
                        straighter

    Returns:
        The ReblockGraph representation, the new portion of the
        optimal road network, and the old portion of the optimal
        road network
    """
    # [1] Bldg poly -> bldg centroids
    bldg_centroids = [b.centroid for b in buildings]

    # [2] Bldgs on block border have access already -- drop them
    bldg_centroids = drop_buildings_intersecting_block(parcels, 
                                                       bldg_centroids,
                                                       block,
                                                       )

    if len(bldg_centroids) == 0:
        warning("Current block contains 0 non-reblocked buildings after dropping already connected buildings")
        return None, None, None 

    # [3] Add dummy bldg centroid to outside of block, forcing Steiner Alg
    #     to connect to the outside road network
    bldg_centroids = add_outside_node(block, bldg_centroids)

    # [4] Convert parcel -> ReblockGraph
    #graph = ReblockGraph.from_multilinestring(parcels)
    graph = ReblockGraph.from_shapely(parcels)
    graph.es['edge_type'] = None

    # [5] Add bldg centroids to ReblockGraph as terminal nodes
    graph.add_buildings(bldg_centroids)

    # [6] Update the edge types to denote existing streets, per block geometry
    # NOTE: when moving from the R parcelization implementation to the 
    # momepy impl the block and parcel coords no longer exactly match
    # up so we need a small snappng function 
    block_snapped = snap_block(block, parcels)
    graph.update_edge_types(block_snapped, check=True)

    # [7 Optional] Add width to ReblockGraph based on bldg geoms
    if use_width or simplify_roads:
        graph.set_edge_width(buildings, simplify=True)
        graph.calc_edge_weight()

    # [8] Clean graph if it's disconnected
    num_components, graph = graph.clean_graph()

    # [10] Steiner Approximation
    graph.steiner_tree_approx()

    # [11] Extract optimal paths from graph and return
    new_path, existing_path = graph.get_steiner_linestrings(expand=False)

    # [12] Optional - add thru streets
    if thru_streets_top_n > 0:
        graph, new_path, existing_path = add_thru_streets(graph, top_n=thru_streets_top_n)

    # [13] Optional - simplify to make streets straighter
    if simplify_roads:
        new_path = simplify_streets(graph) 
    
    return graph, new_path, existing_path

def add_thru_streets(
    graph: ReblockGraph,
    top_n: int = 0,
    ratio_cutoff: Optional[float]=None,
    cost_fn: Optional[Callable[[igraph.Edge], float]]=None,
    ) -> Tuple[ReblockGraph, MultiLineString, MultiLineString]:
    """
    The reblocking algorithm, by definition, tends to find
    trees which then leads to dead-ends in the streets. So,
    this adds thru streets via certain criteria. It sorts dead-ends
    by the ratio of path length post-reblocking and path length
    over the parcel boundaries (i.e. min possible path). Then, 
    the user can select to add the top_n most severe paths, or
    they can select to add all those over a certain threshold

    Args:
        graph: planar graph containing reblocking representation
        top_n: if provided, will connect the top_n most severe dead-ends
        ratio_cutoff: if provided, will connect all those dead-ends
                      with severity over the threshold
        cost_fn: defaults to calculating path length via Euclidean 
                 distance but provides user functionality for custom
                 cost function
    Returns:
        Replicates the return from reblocking, so function can 
        be a seamless post-processing step
        Graph, new roads, existing roads
    """
    graph.add_through_lines(top_n=top_n,
                            ratio_cutoff=ratio_cutoff,
                            cost_fn=cost_fn,
                            )
    new_path, existing_path = graph.get_steiner_linestrings(expand=False)
    return graph, new_path, existing_path

def simplify_streets(
    graph: ReblockGraph,
    ) -> Tuple[ReblockGraph, MultiLineString, MultiLineString]:
    """
    The parcelization process results in candidates that are not
    representative of real life roads -- their shape is weird.
    This post-processing function smplifies the street network st
    it reduces the number of turns in between given points, thus
    making the road network straighter and simpler. It will do this
    while respecting the existing road network, so all simplified
    roads will remain viable.
    """
    
    simplified_linestrings = graph.simplify_reblocked_graph()
    return simplified_linestrings

def main(buildings_path: Union[Path, str], 
         parcels_path: Union[Path, str], 
         blocks_path: Union[Path, str], 
         output_dir: Union[Path, str],
         overwrite: bool = False,
         use_width: bool = False, 
         simplify_roads: bool = False,
         thru_streets_top_n: int = 0,
         progress: bool = True,
         block_list: Optional[List[str]] = None,
         ) -> None:
    """
    Reblock given paths to buildings, parcels, and blocks,
    and an output directory.
    Allows for reblocking options including using width 
    in Steiner Approximation, adding thru streets to reduce
    dead-ends in the reblocking, and simplifying the streets
    to make a straigther reblocking. Further, exposes options
    for only reblocking specific block_id's rather than all
    blocks in the input files, for displaying progress, and 
    for overwriting existing work.

    Args:
        buildings_path: gadm files for building polygons
        parcesl_path: gadm file for parcel polys or linestrings
        blocks_path: gadm file for existing road networks (i.e. blocks)
        output_dir: directory to save output reblocking file
        overwrite: if True, will save over work if output file exists
        use_width: if True, calculates weights as w = distance/width
        thru_streets_top_n: if not None, will add thru streets to
                            mitigated dead ends resulting from reblocking.
                            Adds the top-k most severe dead ends
        simplify_roads: if True, does post-processing to make roads
                        straighter
        progress: if False, will not output block-by-block progress
        block_list: Optionally the user can specify only some of
                    the blocks to reblock, rather than doing all of them.
                    Also allows for priority of reblocking

    """

    buildings_path = Path(buildings_path)
    parcels_path = Path(parcels_path)
    blocks_path = Path(blocks_path)
    output_dir = Path(output_dir)

    gadm = buildings_path.stem.replace("buildings_","")
    fname = "reblock_{}".format(gadm)
    if use_width:
        fname += "-width"
    if thru_streets_top_n:
        fname += '-thru{}'.format(thru_streets_top_n)
    if simplify_roads:
        fname += "-simplify"
    output_path = output_dir / (fname + ".geojson")
    
    if (not output_path.is_file()) or overwrite:
        info("Saving reblocking to: %s", output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        bldgs_gdf = gpd.read_file(buildings_path)
        parcels_gdf = gpd.read_file(parcels_path)
        blocks_gdf = csv_to_geo(blocks_path).rename(columns={'block_geom': 'geometry'})
        if block_list is None:
            block_list = blocks_gdf['block_id']
        bldgs_gdf = gpd.sjoin(bldgs_gdf, blocks_gdf, how='left', op='intersects').drop(columns=['index_right'])

        new_roads = {}
        existing_roads = {}

        # Reblock each block ID independentally
        for block_id in (tqdm.tqdm(block_list) if progress else block_list):
            parcels = parcels_gdf[parcels_gdf['block_id']==block_id]['geometry']
            block = blocks_gdf[blocks_gdf['block_id']==block_id]['geometry'].iloc[0]
            bldgs = bldgs_gdf[bldgs_gdf['block_id']==block_id]['geometry']
            if bldgs.shape[0] > 1:
                bldgs = bldgs.values
                parcels = parcels.explode()

                _, new, existing = reblock(parcels, bldgs, block,
                                           use_width=use_width,
                                           simplify_roads=simplify_roads,
                                           thru_streets_top_n=thru_streets_top_n,
                                           )
                if new is not None:
                    new_roads[block_id] = ['new', new]
                if existing is not None:
                    existing_roads[block_id] = ['existing', existing]
        
        new_roads = gpd.GeoDataFrame.from_dict(new_roads, 
                                               orient='index',
                                               columns=['road_type', 'geometry'],
                                               ).reset_index()
        existing_roads = gpd.GeoDataFrame.from_dict(existing_roads, 
                                                    orient='index',
                                                    columns=['road_type', 'geometry'],
                                                    ).reset_index()
        reblock_gdf = pd.concat([new_roads, existing_roads])
        reblock_gdf.rename(columns={'index': 'block_id'}, inplace=True)

        reblock_gdf = reblock_gdf.loc[~reblock_gdf['geometry'].isna()]
        reblock_gdf.to_file(output_path, driver='GeoJSON')
    else:
        info("Reblocking exists already at: %s", output_path)


