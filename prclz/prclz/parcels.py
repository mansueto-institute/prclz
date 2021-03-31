import geopandas as gpd
import pandas as pd
from shapely.geometry import (LinearRing, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.ops import nearest_points
import momepy
import numpy as np
from typing import List, Tuple, Union, Sequence
from pathlib import Path
from logging import warning, info
#from ..utils import csv_to_geo
import argparse
from prclz.utils import csv_to_geo

def make_parcels(bldgs: Union[Sequence[Polygon], MultiPolygon],
                 block: Polygon,
                 ) -> gpd.GeoDataFrame:
    """
    Breaks block polygon in parcels based on bldgs. Merges orphaned
    parcels resulting from convex Voronoi. If the bldg count == 0, 
    then parcelization is trivial and we just return the block

    Args:
        bldgs: building polygons
        block: block polygon

    Returns:
        Dataframe w/ polygons as parcels
    """
    if len(bldgs) == 0:
        parcels_gdf = gpd.GeoDataFrame({'geometry': [block]}, crs='EPSG:4326')
    else:
        # Change CRS
        block = gpd.GeoSeries(block, crs='EPSG:4326').to_crs("EPSG:3395")[0]
        bldgs = bldgs.to_crs("EPSG:3395")   

        # Basic tesselation
        tess_gdf, bldgs_gdf = tessellate(bldgs, block)

        # Split resulting tess by those w/ or w/o a building
        no_bldg_gdf, has_bldg_gdf = get_orphaned_polys(tess_gdf, bldgs_gdf)

        # Reunion
        parcels_gdf = reunion(no_bldg_gdf, has_bldg_gdf, bldgs_gdf)
        #parcels_gdf = tess_gdf

        # Revert CRS back
        parcels_gdf = parcels_gdf.to_crs("EPSG:4326")

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

def get_orphaned_polys(tessellations: gpd.GeoDataFrame,
                       bldgs: gpd.GeoDataFrame,
                       ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    The Morphological Tess from Momepy probably has orphaned parcels
    in it resulting from the non-convex block. This splits the parcels
    into those w/ a building and those w/o a building (i.e. orphaned)

    Args:
        tessellations: output dataframe from momepy.Tessellation
        bldgs: dataframe of building polygons

    Returns:
        Two dataframes of parcels w/o and w/ buildings, respectively
        Geometry type for each parcel is polygon (never multipolygon)
    """
    # Get the multi polys
    t = tessellations.copy()
    t['is_mp'] = t.type.isin(["MultiPolygon"])
    tess_multips = t[t['is_mp']]
    tess_multips = tess_multips.explode()

    # Sjoin against buildings
    tess_multips = gpd.sjoin(tess_multips, bldgs, how='left', op='intersects')

    # Keep only those w/o building
    orphan_idx = tess_multips['index_right'].isna()
    no_bldg = tess_multips[orphan_idx]
    has_bldg = tess_multips[~orphan_idx]

    # Add back the earlier polygons for completeness
    no_bldg = no_bldg[['geometry']]
    has_bldg = has_bldg[['uID_left','geometry']].rename(columns={'uID_left': 'uID'})
    orig_bldg = t[~t['is_mp']][['uID','geometry']]
    has_bldg = pd.concat([has_bldg, orig_bldg])
    has_bldg.reset_index(drop=True, inplace=True)
    no_bldg.reset_index(drop=True, inplace=True)

    return no_bldg, has_bldg

def find_parent_parcel_id(parcel: Polygon,
                       parents: gpd.GeoDataFrame,
                       bldgs: gpd.GeoDataFrame,
                       ) -> Tuple[int, Polygon]:
    """
    Given a single parcel and dataframes of candidate parent parcels
    and buildings, finds the parcel from parents that:
        1. Neighbors the target parcel geometry
        2. Minimizes dist btwn the parcel building and target parcel centroid
    NOTE: each dataframe should have 'uID' id column

    Args:
        parcel: target parcel geometry
        parents: parent geometries to match target parcel to
        bldgs: buildings dataframe used to get dist to target parcel

    Returns:
        The geometry from parents AND the index from 'uID'
    """
    assert 'uID' in parents.columns, "ERROR -- in find_parent_parcel parents should include 'uID'"
    assert 'uID' in bldgs.columns, "ERROR -- in find_parent_parcel bldgs should include 'uID'"
    
    # Sort buildings by distance to parcel centroid
    centroid = parcel.centroid
    bldgs = bldgs.copy()
    bldgs['dist'] = bldgs.distance(centroid)
    bldgs.sort_values(by='dist', ascending=True, inplace=True)

    # Starting w/ nearest, loop until we find a bldg
    # where the line from parent_candidate_centroid -> centroid 
    # is entirely within the block
    found_match = False
    for bid in bldgs['uID'].values:
        parent = parents[parents['uID']==bid].iloc[0]['geometry']
        nearest_pt = nearest_points(parent, parcel)
        if nearest_pt[0] == nearest_pt[1]: # implies parcels border each other
            found_match = True
            break
    if not found_match:
        bid = None 
        # NOTE: this does not imply failure, as many orphaned polys
        # are in areas w/o any buildings at all
        info("Could not match orphaned parcel with centroid %s to neighboring parcel", centroid)

    return bid

def reunion(no_bldg: gpd.GeoDataFrame,
            has_bldg: gpd.GeoDataFrame,
            bldgs_df: gpd.GeoDataFrame,
            ) -> gpd.GeoDataFrame:
    """
    Map each orphaned parcel in no_bldg to the proper parent
    parcel in has_bldg, using the uID field to map buildings
    to parcels.
    """
    reunioned = no_bldg.copy()
    reunioned['uID'] = [find_parent_parcel_id(orphan, has_bldg, bldgs_df) 
                        for orphan in reunioned['geometry']
                        ]
    reunioned = pd.concat([reunioned, has_bldg])
    reunioned = reunioned.dissolve(by='uID')
    reunioned.reset_index(inplace=True)

    return reunioned


def main(
    blocks_path: Union[Path, str], 
    buildings_path: Union[Path, str], 
    output_dir: Union[Path, str], 
    overwrite: bool = False,
    ) -> None:
    """
    Given a block geojson file and a buildings geojson file, and an
    output directory, creates the corresponding parcels file 
    """
    blocks_path = Path(blocks_path)
    buildings_path = Path(buildings_path)
    output_dir = Path(output_dir)
    
    gadm = blocks_path.stem.replace("blocks_","")
    output_path = output_dir / "parcels_{}.geojson".format(gadm)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if (not output_path.is_file() ) or overwrite:

        blocks_gdf = csv_to_geo(blocks_path).rename(columns={'block_geom': 'geometry'})
        blocks_gdf.crs = "EPSG:4326"
        blocks_gdf.geometry.crs = "EPSG:4326"
        start_crs = blocks_gdf.crs
        bldgs_gdf = gpd.read_file(buildings_path)

        # Map each bldg to block_id
        bldgs_gdf = gpd.sjoin(bldgs_gdf, blocks_gdf, how='left', op='intersects')

        # Parcelize each block_id -- this could be parellelized if
        # performance becomes a bottleneck
        all_parcels = []
        for i, block_id in enumerate(blocks_gdf['block_id']):        
            bldgs = bldgs_gdf[bldgs_gdf['block_id']==block_id]['geometry']
            block = blocks_gdf[blocks_gdf['block_id']==block_id]['geometry'].iloc[0]
            
            parcels_gdf = make_parcels(bldgs, block)
            parcels_gdf['block_id'] = block_id
            all_parcels.append(parcels_gdf)

        parcels_gdf = pd.concat(all_parcels)
        parcels_gdf.drop(columns=['uID'], inplace=True)
        parcels_gdf.to_file(output_path, driver='GeoJSON')
        info("Save gadm %s parcels at %s", gadm, output_path)
    else:
        info("Gadm %s parcels already exist at %s", gadm, output_path)


def check_within(
    parcels_gdf: gpd.GeoDataFrame,
    bldgs_gdf: gpd.GeoDataFrame,
    ) -> None:
    """
    Each parcel should only fully contain 0 or 1 buildings.
    If there are two buildings fully within a parcel, that's bad.
    This function is not directly active anywhere but may be useful
    for debugging given this implementation of parcelization is
    considerably less tested than our other code
    """
    bldgs_gdf['bID'] = np.arange(bldgs_gdf.shape[0])
    parcels_gdf['pID'] = np.arange(parcels_gdf.shape[0])
    contained = gpd.sjoin(parcels_gdf[['geometry','pID']], 
                          bldgs_gdf[['geometry', 'bID']], 
                          how='left', op='contains')
    contained['has_bldg'] = contained['bID'].notna()
    gb = contained.groupby('pID').sum()['has_bldg'].value_counts()
    max_contain = gb.index.max()
    assert max_contain <= 1, "ERROR - there are {} parcels containing {} buildings".format(max_contain, gb[max_contain])

def get_bad_geoms(
    parcels_gdf: gpd.GeoDataFrame,
    bldgs_gdf: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Another utility for debugging/exploring/QC'ing the parcelization output.
    There are instances where our parcels intersect more than one building
    and this function identifies the 'bad' parcels and corresponding buildings.
    From visual inspection we observe that parcels intersecting > 1 building is
    due to slight imprecision when buildings are essentially touching
    """
    bldgs_gdf['bID'] = np.arange(bldgs_gdf.shape[0])
    parcels_gdf['pID'] = np.arange(parcels_gdf.shape[0])
    intersects = gpd.sjoin(parcels_gdf[['geometry','pID']], 
                          bldgs_gdf[['geometry', 'bID']], 
                          how='left', op='intersects')
    intersects['has_bldg'] = intersects['bID'].notna()
    bldg_counts = intersects.groupby('pID').sum()
    bad_pID = bldg_counts[bldg_counts['has_bldg']>1].index.values

    parcels_gdf.set_index('pID', inplace=True)
    bad_parcels = parcels_gdf.iloc[bad_pID]
    bad_bldgs = gpd.sjoin(bldgs_gdf[['geometry', 'bID']],
                          bad_parcels.reset_index()[['geometry','pID']],
                          how='inner', op='intersects'
                          )
    return bad_parcels, bad_bldgs


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("blocks_path", type = str)
    parser.add_argument("buildings_path", type = str)
    parser.add_argument("output_dir",     type = str)
    parser.add_argument("--overwrite", help = "overwrite existing files", action='store_true')
    args = parser.parse_args()
    main(args.blocks_path, args.buildings_path, args.output_dir, args.overwrite)
