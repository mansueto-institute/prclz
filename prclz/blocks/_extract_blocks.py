import datetime
import logging
from logging import error, info, log, warning
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import pandas as pd
import psutil
import shapely.wkt
from prclz.blocks._methods import BufferedLineDifference
from prclz.utils import get_gadm_level_column
from psutil._common import bytes2human
from shapely.geometry import MultiPolygon, Polygon


def log_memory_info(index, logger):
    mem = psutil.virtual_memory()
    mem_info = ", ".join(['%s: %s' % (name, (lambda value: bytes2human(value) if value != 'percent' else value)(getattr(mem, name))) for name in mem._fields])
    logger.info("memory usage for %s: %s", index, mem_info)

def read_file(path, **kwargs):
    """ ensures geometry set correctly when reading from csv
    otherwise, pd.BlockManager malformed when using gpd.read_file(*) """
    if not path.endswith(".csv"):
        return gpd.read_file(path)
    raw = pd.read_csv(path, **kwargs)
    raw["geometry"] = raw["geometry"].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(raw, geometry="geometry")

def extract(index: str, geometry: Union[Polygon, MultiPolygon], linestrings: gpd.GeoDataFrame, target: Path, output_dir: Path) -> None:
    info("Running extraction for %s", index)
    log_memory_info(index, logging.getLogger())
    block_polygons = BufferedLineDifference().extract(geometry, linestrings.unary_union)
    blocks = gpd.GeoDataFrame(
        [(index + "_" + str(i), polygon) for (i, polygon) in enumerate(block_polygons)], 
        columns=["block_id", "geometry"])
    blocks.set_index("block_id")
    blocks.to_csv(target)
    info("Serialized blocks from %s to %s", index, target)
    log_memory_info(index, logging.getLogger())

def main(gadm_path: Path, linestrings_path: Path, output_dir: Path, gadm_level: int, overwrite: bool):
    timestamp = datetime.datetime.now().isoformat()
    logger = logging.getLogger()
    log_dst = output_dir/"logs/blocks"
    log_dst.mkdir(exist_ok = True, parents = True)
    logger.addHandler(logging.FileHandler(log_dst/f"{gadm_path.stem}_{timestamp}.log"))

    index = gadm_path.stem
    filename = output_dir/f"blocks_{index}.csv"

    if (not filename.exists()) or (filename.exists() and overwrite):
        info("Reading geospatial data from files.")
        log_memory_info("main", logger)
        gadm        = read_file(str(gadm_path))
        linestrings = gpd.read_file(str(linestrings_path))

        info("Setting up indices.")
        log_memory_info("main", logger)
        gadm_column, gadm_level = get_gadm_level_column(gadm, gadm_level)
        gadm = gadm.set_index(gadm_column) 

        info("Overlaying GADM boundaries on linestrings.")
        log_memory_info("main", logger)
        overlay = gpd.sjoin(gadm, linestrings, how="left", op="intersects")\
                    .groupby(lambda idx: idx)["index_right"]\
                    .agg(list)

        info("Aggregating linestrings by GADM-%s delineation.", gadm_level)
        log_memory_info("main", logger)
        gadm_aggregation = gadm.join(overlay)[["geometry", "index_right"]]\
                            .rename({"index_right": "linestring_index"}, axis=1)

        extractor = BufferedLineDifference()
        info("Extracting blocks for each delineation using method: %s.", extractor)
        log_memory_info("main", logger)
        for (index, geometry, ls_idx) in gadm_aggregation.itertuples():
            try:
                extract(index, geometry, linestrings.iloc[ls_idx], filename, output_dir) 
            except Exception as e:
                error("%s while processing %s: %s", type(e).__name__, index, e)
                with open(output_dir/f"error_{index}", 'a') as error_file:
                    print(e, file=error_file)
        log_memory_info("main", logger)
    else:
        info("Skipping %s (file %s exists and no overwrite flag given)", index, filename)
    info("Done.")