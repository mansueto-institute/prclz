from typing import List, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, MultiLineString
from shapely.ops import cascaded_union
from shapely.wkt import loads
import pandas as pd
import time 

import os 
import matplotlib.pyplot as plt 
import sys 

import argparse

from .setup_paths import build_data_dir, TRANS_TABLE

def geofabrik_to_gadm(geofabrik_name):
    country_info = TRANS_TABLE[TRANS_TABLE['geofabrik_name'] == geofabrik_name]

    if country_info.shape[0] != 1: 
        print("geofabrik -> gadm failed, CHECK csv mapping for {}".format(geofabrik_name))
        return None, None 

    gadm_name = country_info['gadm_name'].iloc[0]
    region = country_info['geofabrik_region'].iloc[0]
    return gadm_name, region.title()


def csv_to_geo(csv_path, add_file_col=False):
    '''
    Loads the csv and returns a GeoDataFrame
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

def join_block_files(block_file_path: str) -> gpd.GeoDataFrame:

    block_files = [f for f in os.listdir(block_file_path) if "csv" in f]

    #return block_files

    all_blocks = pd.concat([csv_to_geo(os.path.join(block_file_path, block_file), add_file_col=True) 
        for block_file in block_files if "error" not in block_file])

    all_blocks = gpd.GeoDataFrame(all_blocks, geometry='block_geom')

    return all_blocks


def split_files_alt(file_name: str, trans_table: pd.DataFrame, return_all_blocks=False, gadm_name=None, region=None):
    '''
    Given a country buildings.geojson file string,
    it distributes those buildings according to the
    GADM boundaries and block file
    '''

    geofabrik_name = file_name.replace("_buildings.geojson", "").replace("_lines.geojson", "")
    
    if gadm_name is None or region is None:
        gadm_name, region = geofabrik_to_gadm(geofabrik_name)
    
    obj_type = "lines" if "lines" in file_name else "buildings"

    # Get the corresponding block file (and check that it exists)
    block_file_path = os.path.join(BLOCK_PATH, region, gadm_name)

    # Check that the block folder exists
    if not os.path.isdir(block_file_path):
        print("WARNING - country {} / {} does not have a block folder".format(geofabrik_name, gadm_name))
        return None, "no_block_folder" 

    # Check that the block folder actually has block files in it
    block_files = os.listdir(block_file_path)

    if len(block_files) == 0:
        print("WARNING - country {} / {} has a block file path but has no block files in it".format(geofabrik_name, gadm_name))
        return None, "block_folder_empty" 

    all_blocks = join_block_files(block_file_path)
    all_blocks.set_index('block_id', inplace=True)
    assert all_blocks.index.is_unique, "Setting index=block_id but not unique"

    # Get buildings or lines file
    try:
        geojson_objects = gpd.read_file(os.path.join(GEOJSON_PATH, region, file_name))
    except:
        print("WARNING - country {} / {} could not load GeoJSON file".format(geofabrik_name, gadm_name))
        return None, "load_geojson_error"

    cols = [c for c in geojson_objects.columns]
    cols.append('gadm_code')
    geojson_objects['match_count'] = 0

    print("Processing a {} file type of country {}\n".format(obj_type, geofabrik_name))

    # Just safely make our output directories, if needed
    if not os.path.isdir(os.path.join(DATA_PATH, obj_type, region)):
        os.mkdir(os.path.join(DATA_PATH, obj_type, region))

    output_path = os.path.join(DATA_PATH, obj_type, region, gadm_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)


    # Identify those geojson_objects that sjoin via intersect with our block
    if obj_type == "buildings":
        geojson_objects['bldg_centroid'] = geojson_objects['geometry'].centroid 
        geojson_objects.set_geometry('bldg_centroid', inplace=True)

        # Check that we are not missing any centroids or geojson_objects
        assert not (geojson_objects['geometry'].isna().any()), "building geometry is missing"
        assert not (geojson_objects['bldg_centroid'].isna().any()), "bldg_centroid is missing"

        # sjoin on all_blocks which then adds 'gadm_code' to our geojson_objects
        geojson_objects = gpd.sjoin(geojson_objects, all_blocks, how='left', op='within')
        assert 'gadm_code' in list(geojson_objects.columns)
        assert isinstance(geojson_objects, gpd.GeoDataFrame), "No longer GeoDataFrame (1)"
        geojson_objects.set_geometry('geometry', inplace=True)
        geojson_objects['match_count'] = pd.notnull(geojson_objects['index_right'])

    elif obj_type == "lines":

        # Check that we are not missing any geometry objects
        assert not (geojson_objects['geometry'].isna().any()), "lines geometry is missing"

        # sjoin on all_blocks which then adds 'gadm_code' to our geojson_objects
        geojson_objects = gpd.sjoin(geojson_objects, all_blocks, how='left', op='intersects')
        assert 'gadm_code' in list(geojson_objects.columns)
        assert isinstance(geojson_objects, gpd.GeoDataFrame), "No longer GeoDataFrame (1)"
        geojson_objects['match_count'] = pd.notnull(geojson_objects['index_right'])

    else:
        assert False, "Hacky error -- check obj_type"

    # Now, distribute and save
    all_gadm_codes = geojson_objects['gadm_code'].unique()
    gadm_code_match_count = {}
    for code in all_gadm_codes:

        # Do nothing for those unmatched (but will note those that are not matched)
        if pd.notna(code):

            geojson_objects_in_gadm = geojson_objects[ geojson_objects['gadm_code']==code ][cols]
            f = obj_type + "_" + code + ".geojson"

            if os.path.isfile(os.path.join(output_path, f)):
                os.remove(os.path.join(output_path, f))

            if geojson_objects_in_gadm.shape[0] > 0:
                geojson_objects_in_gadm.to_file(os.path.join(output_path, f), driver='GeoJSON')

            gadm_code_match_count[code] = geojson_objects_in_gadm.shape[0]

    geojson_objects['match_count'] = geojson_objects['match_count'].apply(int)

    # Update our gadm_code_match_count to include those gadm areas with NO geojson_objects
    for code in list(all_blocks['gadm_code']):
        if code not in gadm_code_match_count:
            gadm_code_match_count[code] = 0

    # And now count how many geojson_objects are unmatched and add that to the counter
    gadm_code_match_count['NO_GADM_DISTRICT'] = geojson_objects[geojson_objects['match_count'] == 0].shape[0]

    count_df = pd.DataFrame.from_dict(gadm_code_match_count, orient='index', columns=['match_count'])
    count_df.reset_index(inplace=True)
    count_df.rename(columns={"index":"gadm_code"}, inplace=True)
    count_df['gadm_name'] = gadm_name

    if return_all_blocks:
        return (geojson_objects, count_df, all_blocks), "DONE"
    else:
        return (geojson_objects, count_df), "DONE"


def check_building_file_already_processed(gadm_name, region, obj_type):
    '''
    Returns True if the gamd_name has already been split out
    Returns False if it needs to be done
    '''

    p = os.path.join(DATA_PATH, obj_type, region, gadm_name)
    if os.path.isdir(p):
        files = os.listdir(p)
        if len(files) > 0:
            return True 
        else:
            return False
    else:
        return False 

def map_matching_results_buildings(buildings_output, all_blocks, file_name):

    nonmatched_pct = (1 - buildings_output.match_count.mean()) * 100
    nonmatched_count = nonmatched_pct * buildings_output.shape[0] / 100

    buildings_output.set_geometry('bldg_centroid', inplace=True)
    ax = buildings_output.plot(column='match_count', figsize=(35,35))
    all_blocks.plot(ax=ax, color='blue', alpha=0.5)
    plt.axis('off')
    plt.title("Nonmatched count = {} or pct = {:.2f}%".format(int(nonmatched_count), nonmatched_pct))
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def map_matching_results_lines(lines_output, all_blocks=None, file_name=None):
    nonmatched_pct = (1 - lines_output.match_count.mean()) * 100
    nonmatched_count = nonmatched_pct * lines_output.shape[0] / 100

    # Add flags
    lines_output['has_natural'] = lines_output['natural'].notnull().map(int)
    lines_output['has_highway'] = lines_output['highway'].notnull().map(int)
    lines_output['has_waterway'] = lines_output['waterway'].notnull().map(int)

    #lines_output.set_geometry('bldg_centroid', inplace=True)

    # Plot those that are matched
    ax = lines_output[ (lines_output.has_natural==1) & (lines_output.match_count==1) ].plot(color='green', figsize=(35,35))
    lines_output[ (lines_output.has_highway==1) & (lines_output.match_count==1)  ].plot(ax=ax, color='black')
    lines_output[ (lines_output.has_waterway==1) & (lines_output.match_count==1)  ].plot(ax=ax, color='blue')

    lines_output[ (lines_output.has_natural==1) & (lines_output.match_count==0) ].plot(ax=ax, color='red')
    lines_output[ (lines_output.has_highway==1) & (lines_output.match_count==0)  ].plot(ax=ax, color='organge')
    lines_output[ (lines_output.has_waterway==1) & (lines_output.match_count==0)  ].plot(ax=ax, color='yellow')

    all_blocks.plot(ax=ax, color='blue', alpha=0.5)
    plt.axis('off')
    plt.title("Nonmatched count = {} or pct = {:.2f}%".format(int(nonmatched_count), nonmatched_pct))
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def main(file_name, REPLACE, gadm_name):

    start = time.time()
    
    geofabrik_name = file_name.replace("_buildings.geojson", "").replace("_lines.geojson", "")

    if gadm_name is None:
        gadm_name, region = geofabrik_to_gadm(geofabrik_name)
        if gadm_name is None and region is None:
            return 
    else:
        country_info = TRANS_TABLE[TRANS_TABLE['gadm_name'] == gadm_name]
        region = country_info['geofabrik_region'].iloc[0].title()

    TYPE = "buildings" if "buildings" in file_name else "lines"

    # Ensure output summary paths exist
    if not os.path.isdir("splitter_output"):
        os.mkdir("splitter_output")
    if not os.path.isdir(os.path.join("splitter_output", TYPE)):
        os.mkdir(os.path.join("splitter_output", TYPE))
    summary_path = os.path.join("splitter_output", TYPE, gadm_name)
    if not os.path.isdir(summary_path):
        os.mkdir(summary_path)

    # Check first if we've already processed the files
    already_processed = check_building_file_already_processed(gadm_name, region, TYPE)

    if already_processed and not REPLACE:
        print("Country {} | {} | {} has already been processed -- SKIPPING".format(geofabrik_name, gadm_name, region))

    else:
        # Do the actual matching and splitting
        rv, details = split_files_alt(file_name, TRANS_TABLE, return_all_blocks=True, gadm_name=gadm_name, region=region)

        # Was a success
        if rv is not None:

            # Unpack
            geojson_objects, count_df, all_blocks = rv 

            # Save out those buildings that didn't match
            not_matched_objects = geojson_objects[geojson_objects['match_count'] == 0]
            nonmatched_path = os.path.join(summary_path, file_name.replace(TYPE, "not_matched_{}".format(TYPE)))

            if os.path.isfile(nonmatched_path):
                os.remove(nonmatched_path)

            if not_matched_objects.shape[0] != 0:
                if 'bldg_centroid' in not_matched_objects.columns:
                    not_matched_objects = not_matched_objects.drop(columns=['bldg_centroid'])
                not_matched_objects.to_file(nonmatched_path, driver='GeoJSON')

            # Save out a png summary of matched/non-matched buildings vs gadm bounds
            matching_viz_path = os.path.join(summary_path, "{}_match.png".format(TYPE))

            if TYPE == "buildings":
                map_matching_results_buildings(geojson_objects, all_blocks, matching_viz_path)
            else:
                map_matching_results_lines(geojson_objects, all_blocks, matching_viz_path)

            # Save out the counts DataFrame
            counts_df_path = os.path.join(summary_path, "matching_counts_summary.csv")
            count_df.to_csv(counts_df_path)

        else:
            error_summary = open(os.path.join("splitter_output", TYPE, "error_summary{}.txt".format(gadm_name)), 'w')
            error_summary.write(file_name + "  |  " + details + "\n")
            error_summary.close()

        # NOTE: add the below to a csv file, along with the matched pcts and counts
        print("Processing {} | {} takes {} seconds".format(geofabrik_name, gadm_name, time.time()-start))



"""
NOTE: our par sbatch script gives a stream of paths to
the buildings.geojson file. This file is unique per country, and
thus also identifies which countries to process
"""

def split_buildings(data_root: str, 
                    gadm_name: str = None,
                    replace: bool = False,
                    ) -> None:

    data_paths = build_data_dir(data_root)
    global TRANS_TABLE    
    
    if gadm_name is None:
        gadm_names = TRANS_TABLE['gadm_name']
    else:
        gadm_names = [gadm_name]

    for gadm_name in gadm_names:
        geofabrik_name = TRANS_TABLE[TRANS_TABLE['gadm_name']==gadm_name]['geofabrik_name'].iloc[0]        
        file_name = geofabrik_name + "_buildings.geojson"
        main(file_name, replace, gadm_name)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--gadm_name", default='All', type=str, 
                        help="The default is to process All the countries, but if you want to process only one country then you can specify a 3-letter GADM country code")
    parser.add_argument("--replace", help="default behavior is to skip if the country has been processed. Adding this option replaces the files",
                         action="store_true")
    
    args = parser.parse_args()

    if args.gadm_name == "All":
        gadm_names = TRANS_TABLE['gadm_name']
    else:
        gadm_names = [args.gadm_name]  

    for gadm_name in gadm_names:
        geofabrik_name = TRANS_TABLE[TRANS_TABLE['gadm_name']==args.gadm_name]['geofabrik_name'].iloc[0]
        file_name = geofabrik_name + "_buildings.geojson"

        main(file_name, args.replace, gadm_name)
        