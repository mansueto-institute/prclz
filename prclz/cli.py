from logging import basicConfig, debug
from pathlib import Path
from typing import Optional, OrderedDict, Sequence, Union

import click

from . import complexity, parcels
from .blocks import extract
from .etl import download as _download
from .etl import split_buildings
from .reblock import reblock

# from https://github.com/pallets/click/issues/513#issuecomment-301046782
class DefinitionOrderGroup(click.Group):
    """Command Group that lists subcommands in the order they were added."""
    def list_commands(self, _):
        """List command names in the order they were specified.
        Assumes Python 3 to leverage dictionary key order reflecting insertion order."""
        return self.commands.keys()

@click.group(cls = DefinitionOrderGroup)
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
    if datasource.lower() not in ["gadm", "geofabrik"]:
        raise click.BadParameter("Datasource must be one of [gadm|geofabrik]")
    _download.main(datasource.lower(), directory, countries.split(",") if countries else countries, overwrite)

@prclz.command()
@click.argument("bldg_file",  type = str, required = True)
@click.argument("gadm_path",  type = str, required = True)
@click.argument("output_dir", type = str, required = True)
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def split_buildings(bldg_file, gadm_path, output_dir):
    """ Split OSM buildings by GADM delineation. """
    split_buildings.main(bldg_file, gadm_path, output_dir)


@prclz.command()
@click.argument("gadm_path",        type = click.Path(exists = True))
@click.argument("linestrings_path", type = click.Path(exists = True))
@click.argument("output_dir",       type = click.Path(exists = True))
@click.option("--gadm_level", help = "GADM aggregation level",  default = 5)
@click.option("--overwrite",  help = "overwrite existing files", default = False, is_flag = True)
def blocks(gadm_path, linestrings_path, output_dir, gadm_level, overwrite):
    """ Define city block geometry. """
    extract.main(Path(gadm_path), Path(linestrings_path), Path(output_dir), gadm_level, overwrite)

@prclz.command()
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def parcels(blocks_path, buildings_path, output_dir, overwrite):
    """ Split block space into cadastral parcels. """
    parcels.main(Path(blocks_path), Path(buildings_path), Path(output_dir), overwrite)

@prclz.command()
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--overwrite", help = "overwrite existing files", default = False, is_flag = True)
def complexity(blocks_path, buildings_path, output_dir, overwrite):
    """ Calculate k-index (complexity) of a city block. """
    complexity.main(Path(blocks_path), Path(buildings_path), Path(output_dir), overwrite)

@prclz.command()
@click.argument("buildings_path", type = click.Path(exists = True))
@click.argument("parcels_path",   type = click.Path(exists = True))
@click.argument("blocks_path",    type = click.Path(exists = True))
@click.argument("output_dir",     type = click.Path(exists = True))
@click.option("--connect_n",      help = "connect n most severe dead ends",    default = 0)
@click.option("--blocks",         help = "specify specific blocks to reblock", default = None)
@click.option("--use_width",      help = "use width for reblocking estimate",  default = False, is_flag = True)
@click.option("--simplify",       help = "simplify reblocked roads",           default = False, is_flag = True)
@click.option("--progress",       help = "display reblocking progress",        default = False, is_flag = True)
@click.option("--overwrite",      help = "overwrite existing files",           default = False, is_flag = True)
def reblock(
    buildings_path: Union[Path, str], 
    parcels_path:   Union[Path, str], 
    blocks_path:    Union[Path, str], 
    output_dir:     Union[Path, str],
    connect_n:      int = 0,
    blocks:         Optional[str] = None,
    simplify:       bool = False,
    use_width:      bool = False, 
    progress:       bool = False,
    overwrite:      bool = False,
    ):
    """ Generate least-cost reblocking network for a city block. """
    reblock.main(
        Path(buildings_path),
        Path(parcels_path),
        Path(blocks_path),
        Path(output_dir),
        overwrite,
        use_width, 
        simplify,
        connect_n,
        progress,
        blocks
    )


if __name__ == '__main__':
    prclz()
