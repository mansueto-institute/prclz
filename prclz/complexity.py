import os
from logging import debug, info
from pathlib import Path
from typing import Sequence, Tuple

import geopandas as gpd
import pandas as pd
import pytess
import shapely
import shapely.wkt
from shapely.geometry import MultiPoint, Point, Polygon

from .topology import PlanarGraph


def get_s0_approximation(block: Polygon, centroids: Sequence[Tuple[float, float]]) -> PlanarGraph:
    """ approximates the initial connectivity graph by partitioning 
    the block into a voronoi decomposition and feeds those faces into
    a planar graph """

    boundary_points = list(block.exterior.coords)
    boundary_set = set(boundary_points)

    # get internal parcels from the voronoi decomposition of space, given building centroids
    intersected_polygons = []
    debug("generating Voronoi decomposition")
    decomposition = pytess.voronoi(centroids)
    debug("intersecting Voronoi decomposition (N=%s) with block geometry", len(decomposition))
    for (anchor, vs) in decomposition:
        if anchor and anchor not in boundary_set and len(vs) > 2:
            anchor_pt = Point(anchor)
            try: 
                polygon = Polygon(vs).buffer(0).intersection(block)
                intersected_polygons.append((anchor_pt, polygon))
            except shapely.errors.TopologicalError as e:
                debug("invalid geometry at polygon %s\n%s", vs, e)

    # simplify geometry when multiple areas intersect original block
    debug("simplifying multi-polygon intersections")
    simplified_polygons = [
        polygon if polygon.type == "Polygon" else next((segment for segment in polygon if segment.contains(anchor)), None)
        for (anchor, polygon) in intersected_polygons]
    
    debug("building planar graph approximation")
    return PlanarGraph.from_polygons([polygon for polygon in simplified_polygons if polygon])

def get_weak_dual_sequence_for_dataframe(
    gdf: gpd.GeoDataFrame, 
    polygon_column: str = "geometry", 
    centroid_column: str = "centroids"
) -> Sequence[PlanarGraph]:
    s_vector = [get_s0_approximation(gdf[polygon_column], [(p.x, p.y) for p in gdf[centroid_column]])]
    while s_vector[-1].number_of_nodes() > 2:
        s_vector.append(s_vector[-1].weak_dual())
    return s_vector

def get_weak_dual_sequence(block: Polygon, centroids: Sequence[Point]) -> Sequence[PlanarGraph]:
    s_vector = [get_s0_approximation(block, [(c.x, c.y) for c in centroids])]
    k = 0
    while s_vector[-1].number_of_nodes() > 0:
        k += 1
        debug("weak dual sequence running... (%s)", k)
        s_vector.append(s_vector[-1].weak_dual())
    s_vector.pop() # last graph has no nodes
    return s_vector

def get_complexity(sequence: Sequence[PlanarGraph]) -> int:
    return len(sequence) - 1 if sequence else 0

def read_file(path, **kwargs):
    """ ensures geometry set correctly when reading from csv
    otherwise, pd.BlockManager malformed when using gpd.read_file(*) """
    if not path.endswith(".csv"):
        return gpd.read_file(path)
    raw = pd.read_csv(path, **kwargs)
    raw["geometry"] = raw["geometry"].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(raw, geometry="geometry")


def calculate_complexity(index, output, block, centroids, cache_files):
    block_cache = output.parent/(str(index) + ".block.cache")
    if block_cache.exists():
        info("Reading cached k-value for %s", index)
        with block_cache.open('r') as f:
            complexity, centroids_multipoint = f.readlines()
            complexity = int(complexity)
            centroids_multipoint = shapely.wkt.loads(centroids_multipoint.strip().replace('"', ''))
    else: 
        info("Calculating k for %s", index)
        sequence = get_weak_dual_sequence(block, centroids)
        complexity = get_complexity(sequence)
        centroids_multipoint = MultiPoint(centroids)
        with block_cache.open('w') as f:
            print(complexity, file=f)
            print(shapely.wkt.dumps(centroids_multipoint), file=f)
    cache_files.append(block_cache)

    return (index, complexity, centroids_multipoint)


def main(blocks_path: Path, buildings_path: Path, complexity_output: Path, overwrite: bool):
    if (not complexity_output.exists()) or (complexity_output.exists() and overwrite):
        info("Reading geospatial data from files.")
        blocks    = read_file(str(blocks_path), index_col="block_id", usecols=["block_id", "geometry"], low_memory=False)
        buildings = read_file(str(buildings_path), low_memory=False)
        buildings["geometry"] = buildings.centroid

        info("Aggregating buildings by street block.")
        block_aggregation = gpd.sjoin(blocks, buildings, how="right", op="intersects")
        block_aggregation = block_aggregation[pd.notnull(block_aggregation["index_left"])].groupby("index_left")["geometry"].agg(list)
        block_aggregation.name = "centroids"
        block_buildings = blocks.join(block_aggregation)
        block_buildings = block_buildings[pd.notnull(block_buildings["centroids"])]

        cache_files = []
        info("Calculating block complexity.")
        complexity = [calculate_complexity(idx, complexity_output, block, centroids, cache_files) for (idx, block, centroids) in block_buildings[["geometry", "centroids"]].itertuples()]
        
        info("Restructuring complexity calculations by block_id index.")
        block_buildings = block_buildings.join(pd.DataFrame(complexity, columns=["block_id", "complexity", "centroids_multipoint"]).set_index("block_id"))

        info("Serializing complexity calculations to %s.", complexity_output)
        block_buildings[['geometry', 'complexity', 'centroids_multipoint']].to_csv(complexity_output)
        # cleanup 
        info("Removing cache files.")
        for cache_file in cache_files:
            os.remove(cache_file)

        info("Done.")

    else: 
        info("Skipping processing %s (output exists and overwrite flag not given)", complexity_output)
