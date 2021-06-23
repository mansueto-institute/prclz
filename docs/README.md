# documentation for `prclz` 

The following are instructions for the command line interface to the `prclz` library. All of these operations are possible in client code that loads `prclz` as a dependency rather than a command line tool.

## 0. installation

We recommend installation from PyPI via `pip` into a virtual environment (managed by `venv` in this example, but compatible with `conda`, `poetry`, running a Docker image, etc.):
```
$ python3 -mvenv venv
$ source ./venv/bin/activate
pip install prclz 
```

Test your installation by running the `prclz` command without any arguments.
```
$ prclz
```

You should see the help prompt:
```
Usage: prclz [OPTIONS] COMMAND [ARGS]...

Options:
  --logging [CRITICAL|ERROR|WARNING|INFO|DEBUG]
                                  set logging level  [default: INFO]
  --help                          Show this message and exit.

Commands:
  download         Download upstream data files.
  split-buildings  Split OSM buildings by GADM delineation.
  blocks           Define city block geometry.
  parcels          Split block space into cadastral parcels.
  complexity       Calculate k-index (complexity) of a city block.
  reblock          Generate least-cost reblocking network for a city block.
```

Given the multitude of geospatial package dependencies requiring linking to system-level C/C++ libraries and the myriad ways different platforms provide those libraries, you may get a warning about binary incompatibilities. One example is below:

``` 
~/project/venv/lib/python3.8/site-packages/geopandas/_compat.py:106: UserWarning: The Shapely GEOS version (3.8.1-CAPI-1.13.3) is incompatible with the GEOS version PyGEOS was compiled with (3.9.0-CAPI-1.16.2). Conversions between both will be slow.
```

In these cases, we recommend installing the conflicting packages from code rather than binary (wheel) sources:
```
pip install prclz --no-binary pygeos,shapely
```

In the example conflict provided, the non-wheel distribution of `shapely` simply links to existing installations of `GEOS` rather than providing on a bundled binary that conflicts with the one bundled by `pygeos`; see [this issue](https://github.com/Toblerity/Shapely/issues/651). You may have to take another approach to resolve other incompatibilities. Please [file a Github issue on the `prclz` repository](https://github.com/mansueto-institute/prclz/issues/new/choose) if you have other problems installing.

## 1. download upstream data (optional)

### standard data layout
The command line interface assumes a standard layout in a root data directory. Within the root directory, the default `download` subcommands will assume the directory structures looks like the following:
```
root
  ├── GADM
  ├── blocks
  ├── buildings
  ├── cache
  ├── complexity
  ├── errors
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  └── parcels 

```

The `prclz download` subcommands will create these subdirectories if they do not exist. If you require a different layout, see the `build_data_dir()` method in the `prclz.etl` submodule.

### Geofabrik
To pull down all data from the Geofabrik mirror of OpenStreetMap to a directory, run:
```
prclz download geofabrik /path/to/output
```

You can also pass in a list of comma-delimited GADM codes to only download some countries, and force the command to overwrite any existing files:
```
prclz download geofabrik /path/to/output --countries SLE,LBR --overwrite
```

### GADM 
The same `download` subcommand is used to download national and subnational administative boundaries from the GADM database.
```
prclz download gadm /path/to/output
```

Like the `geofabrik` datasource, the `gadm` datasource supports filtering and overwrite options:
```
prclz download gadm /path/to/output --countries SLE,LBR --overwrite
```

## 2. preprocess (optional)

### select building footprints and road networks from Geofabrik layers

The relevant geometry for this workflow resides in the `lines`, `polygons`, and `multipolygons` layers of the Geofabrik downloaded file. The relevant operations to extract these features are expressed most succintly with GDAL file format conversion tools; we provide a convenience shell script in the `/scripts` directory to replicate our OpenStreetMap workflow:

```
cd scripts
./extract.sh /path/to/geofabrik/file /path/to/output
```

### split by GADM 

Additionally, the building footprints need to assigned to GADMs in order to be enable parallel processing of `prclz` functions at country-scale. 

```
prclz splitbuildings /geofabrik/buildings.geojson /gadm/country-boundaries.shp /path/to/output
```

## 3. determine city block geometry from road network boundaries 
To determine city block delineations, pass for each administrative unit, pass in the administrative unit boundaries, and the national road network.

```
prclz blocks /path/to/gadm /path/to/linestrings /path/to/output
```

## 4. calculate block complexity 
To calculate the city block complexity measure, pass in the city block geometry and building footprints.
```
prclz complexity /path/to/block-geometry /path/to/building-footprints /path/to/output
```

## 5. determine cadastral parcel geometries
To break up a city block into its constituent parcels, pass in the city block geometry and building footprints.
```
prclz parcels /path/to/block-geometry /path/to/building-footprints /path/to/output
```

## 6. create optimal access road network proposal
To create the optimal reblocking road network, pass in the building footprints, the parcel boundaries, and city block geometries.
```
prclz reblock /path/to/building-footprints /path/to/parcel-boundaries /path/to/block-geometry /path/to/output
```
