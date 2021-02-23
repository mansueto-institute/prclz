import geopandas as gpd
import pandas as pd
from shapely.geometry import (LinearRing, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.ops import nearest_points
import momepy
import numpy as np
from typing import List, Tuple
from logging import warning

def make_parcels(bldgs: MultiPolygon,
                 block: Polygon,
                 ) -> gpd.GeoDataFrame:
    """
    Breaks block polygon in parcels based on bldgs. Merges orphaned
    parcels resulting from convex Voronoi.

    Args:
        bldgs: building polygons
        block: block polygon

    Returns:
        Dataframe w/ polygons as parcels
    """
    # Basic tesselation
    tess_gdf, bldgs_gdf = tessellate(bldgs, block)

    # Split resulting tess by those w/ or w/o a building
    no_bldg_gdf, has_bldg_gdf = get_orphaned_polys(tess_gdf, bldgs_gdf)

    # Reunion
    parcels_gdf = reunion(no_bldg_gdf, has_bldg_gdf, bldgs_gdf)

    return parcels_gdf

def tessellate(
    bldgs: MultiPolygon,
    block: Polygon,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """ Use momepy to do basic tesselation """
     
    bldgs_gdf = gpd.GeoDataFrame({'geometry':bldgs})
    bldgs_gdf['uID'] = np.arange(bldgs_gdf.shape[0])
    tess_gdf = momepy.Tessellation(bldgs_gdf, unique_id='uID', limit=block).tessellation

    return tess_gdf, bldgs_gdf 

def get_orphaned_polys(tess_gdf: gpd.GeoDataFrame,
                       bldgs_gdf: gpd.GeoDataFrame,
                       ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    The Morphological Tess from Momepy probably has orphaned parcels
    in it resulting from the non-convex block. This splits the parcels
    into those w/ a building and those w/o a building (i.e. orphaned)

    Args:
        tess_gdf: output dataframe from momepy.Tessellation
        bldgs_gdf: dataframe of building polygons

    Returns:
        Two dataframes of parcels w/o and w/ buildings, respectively
        Geometry type for each parcel is polygon (never multipolygon)
    """
    # Get the multi polys
    t = tess_gdf.copy()
    t['is_mp'] = t.type.isin(["MultiPolygon"])
    t_mp = t[t['is_mp']]
    t_mp = t_mp.explode()

    # Sjoin against buildings
    t_mp = gpd.sjoin(t_mp, bldgs_gdf, how='left', op='intersects')

    # Keep only those w/o building
    no_bldg = t_mp[t_mp['index_right'].isna()]
    has_bldg = t_mp[t_mp['index_right'].notnull()]

    # Add back the earlier polygons for completeness
    no_bldg = no_bldg[['geometry']]
    has_bldg = has_bldg[['uID_left','geometry']].rename(columns={'uID_left': 'uID'})
    orig_bldg = t[~t['is_mp']][['uID','geometry']]
    has_bldg = pd.concat([has_bldg, orig_bldg])
    has_bldg.reset_index(drop=True, inplace=True)
    no_bldg.reset_index(drop=True, inplace=True)

    return no_bldg, has_bldg

def find_parent_parcel(parcel: Polygon,
                       parents_df: gpd.GeoDataFrame,
                       bldgs_df: gpd.GeoDataFrame,
                       ) -> Tuple[int, Polygon]:
    """
    Given a single parcel and dataframes of candidate parent parcels
    and buildings, finds the parcel from parents_df that:
        1. Neighbors the target parcel geometry
        2. Minimizes dist btwn the parcel building and target parcel centroid
    NOTE: each dataframe should have 'uID' id column

    Args:
        parcel: target parcel geometry
        parents_df: parent geometries to match target parcel to
        bldgs_df: buildings dataframe used to get dist to target parcel

    Returns:
        The geometry from parents_df AND the index from 'uID'
    """
    assert 'uID' in parents_df.columns, "ERROR -- in find_parent_parcel parents_df should include 'uID'"
    assert 'uID' in bldgs_df.columns, "ERROR -- in find_parent_parcel bldgs_df should include 'uID'"
    
    # Sort buildings by distance to parcel centroid
    centroid = parcel.centroid
    bldgs_df = bldgs_df.copy()
    bldgs_df['dist'] = bldgs_df.distance(centroid)
    bldgs_df.sort_values(by='dist', ascending=True, inplace=True)

    # Starting w/ nearest, loop until we find a bldg
    # where the line from parent_candidate_centroid -> centroid 
    # is entirely within the block
    found_match = False
    for bid in bldgs_df['uID'].values:
        parent = parents_df[parents_df['uID']==bid].iloc[0]['geometry']
        nearest_pt = nearest_points(parent, parcel)
        if nearest_pt[0] == nearest_pt[1]: # implies parcels border each other
            found_match = True
            break
    if not found_match:
        warning("Could not match orphaned parcel to neighboring parcel")

    return bid, parent

def reunion(no_bldg: gpd.GeoDataFrame,
            has_bldg: gpd.GeoDataFrame,
            bldgs_df: gpd.GeoDataFrame,
            ) -> gpd.GeoDataFrame:
    """
    Map each orphaned parcel in no_bldg to the proper parent
    parcel in has_bldg, using the uID field to map buildings
    to parcels.
    """
    reun = no_bldg.copy()
    orphan_uID = []
    for orphan in reun['geometry']:
        bid, _ = find_parent_parcel(orphan, has_bldg, bldgs_df)
        orphan_uID.append(bid)
    reun['uID'] = orphan_uID
    reun = pd.concat([reun, has_bldg])
    reun = reun.dissolve(by='uID')
    reun.reset_index(inplace=True)

    return reun
