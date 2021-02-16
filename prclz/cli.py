from logging import basicConfig, info, debug

import click

from etl import download

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
@click.argument("datasource", type = click.Choice(["gadm", "geofabrik"], case_sensitive = False)) 
@click.argument("directory",  type = click.Path(exists = True))  
@click.argument("countries",  default = None, required = False, nargs = -1)
@click.option("--overwrite",  help = "overwrite existing files", default = False, is_flag = True)
def download(datasource, directory, countries, overwrite):
    """ Download upstream data files. """
    download.main(datasource, directory, countries, overwrite)

@prclz.command()
def split_geojson():
    """ Split OSM data by GADM delineation. """
    pass 

@prclz.command()
def blocks():
    """ Extract block geometry. """
    pass 

@prclz.command()
def parcels():
    """ Split block into cadastral parcels. """
    pass 

@prclz.command()
def complexity():
    """ Calculate the k-index (complexity) of a block. """
    pass 

@prclz.command()
def reblock():
    """ Generate least-cost reblocking network. """
    pass 

if __name__ == '__main__':
    prclz()
