from logging import basicConfig, info, debug

import click

from .etl import download, split_bldgs
from .blocks import extract
from . import complexity

@click.group()
@click.option("--logging", 
    default = "INFO", 
    help = "set logging level", 
    type = click.Choice([ "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]), 
    show_default = True)
def prclz(logging):
    basicConfig(level = logging)

@prclz.command()
def test_logging():
    info("info")
    debug("debug")

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
def split_geojson():
    """ Split OSM data by GADM delineation. """
    pass 

@prclz.command()
@click.argument("--bldg_file", type=str, required=True, help="Path to master geojson file containing all building polygons")
@click.argument("--gadm_path", type=str, required=True, help="Path to GADM file")
@click.argument("--output_dir", type=str, required=True, help="Output directory for gadm-specific building files")
def split_geojson_bldgs(bldg_file, gadm_path, output_dir):
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
def parcels():
    """ Split block into cadastral parcels. """
    pass 

@prclz.command()
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def complexity(blocks_path, buildings_path, output_dir, overwrite):
    """ Calculate the k-index (complexity) of a block. """
    complexity.main(blocks_path, buildings_path, output_dir, overwrite)

@prclz.command()
def reblock():
    """ Generate least-cost reblocking network. """
    pass 

if __name__ == '__main__':
    prclz()
