from logging import basicConfig
from pathlib import Path
from typing import Optional, Sequence, Union

import click

from . import complexity, parcels
from .blocks import extract
from .etl import download, split_bldgs
from .reblock import reblock


@click.group()
@click.option("--logging", 
    default = "INFO", 
    help = "set logging level", 
    type = click.Choice([ "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]), 
    show_default = True)
def prclz(logging):
    basicConfig(level = logging)

@prclz.command()
@click.argument("datasource", type = str, nargs = 1) 
@click.argument("directory",  type = click.Path(exists = True))  
@click.option("--countries", help = "comma-delimited list of GADM codes to filter", default = None, required = False)
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def download(datasource, directory, countries, overwrite):
    """ Download upstream data files. """
    click.echo((datasource, directory, countries, overwrite))
    if datasource.lower() not in ["gadm", "geofabrik"]:
        raise click.BadParameter("Datasource must be one of [gadm|geofabrik]")
    download.main(datasource.lower(), directory, countries.split(",") if countries else countries, overwrite)

@prclz.command()
@click.option("--bldg_file", type=str, required=True, help="Path to master geojson file containing all building polygons")
@click.option("--gadm_path", type=str, required=True, help="Path to GADM file")
@click.option("--output_dir", type=str, required=True, help="Output directory for gadm-specific building files")
def split_geojson(bldg_file, gadm_path, output_dir):
    """ Split OSM buildings by GADM delineation. """
    split_bldgs.main(bldg_file, gadm_path, output_dir)


@prclz.command()
@click.argument("gadm_path",        type = click.Path(exists = True))
@click.argument("linestrings_path", type = click.Path(exists = True))
@click.argument("output_dir",       type = click.Path(exists = True))
@click.option("--gadm_level", help = "GADM aggregation level",   default = 5)
@click.option("--overwrite",  help = "overwrite existing files", default = False, is_flag = True)
def blocks(gadm_path, linestrings_path, output_dir, gadm_level, overwrite):
    """ Extract block geometry. """
    extract.main(gadm_path, linestrings_path, output_dir, gadm_level, overwrite)

@prclz.command()
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def parcels(blocks_path, buildings_path, output_dir, overwrite):
    """ Split block into cadastral parcels. """
    parcels.main(blocks_path, buildings_path, output_dir, overwrite)

@prclz.command()
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def complexity(blocks_path, buildings_path, output_dir, overwrite):
    """ Calculate the k-index (complexity) of a block. """
    complexity.main(blocks_path, buildings_path, output_dir, overwrite)

@prclz.command()
@click.argument("buildings_path",    type = click.Path(exists = True))
@click.argument("parcels_path", type = click.Path(exists = True))
@click.argument("blocks_path",     type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
@click.option("--use_width", help = "use width for reblocking estimate", default = False, is_flag = True)
@click.option("--simplify_roads", help = "simplify reblocked roads", default = False, is_flag = True)
@click.option("--thru_streets_top_k", help = "connect top-k severe dead ends", default = False, is_flag = True)
@click.option("--no_progress", help = "don't display reblocking progress", default = True, is_flag = True)
@click.option("--block_list", help = "specify specific blocks to reblock", default = None)
def reblock(
    buildings_path: Union[Path, str], 
    parcels_path: Union[Path, str], 
    blocks_path: Union[Path, str], 
    output_dir: Union[Path, str],
    overwrite: bool = False,
    use_width: bool = False, 
    simplify_roads: bool = False,
    thru_streets_top_k: Optional[int] = None,
    no_progress: bool = True,
    block_list: Optional[Sequence[str]] = None,
    ):
    """ Generate least-cost reblocking network for analyzed block. """
    progress = not no_progress
    reblock.main(buildings_path, 
                 parcels_path, 
                 blocks_path, 
                 output_dir,
                 overwrite,
                 use_width, 
                 simplify_roads,
                 thru_streets_top_k,
                 progress,
                 block_list,
                 )


if __name__ == '__main__':
    prclz()
