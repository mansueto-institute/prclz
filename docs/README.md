# documentation for `prclz` 

The following are instructions for the command line interface to the `prclz` library. All of these operations are possible in client code that loads `prclz` as a dependency rather than a command line tool.

## 0. installation

We recommend installation from PyPI via `pip` into a virtual environment (managed by `venv` in this example, but compatible with `conda`, `poetry`, running a Docker image, etc.). We have found that the easiest way to use this package is to create a miniconda environment and installing it using:

```
$ conda create --name prclz_env
$ conda activate prclz_env
$ pip install prclz
$ conda install -c conda-forge gdal
```

On Midway, run:
```
module avail python
module load python/anaconda-2020.02
conda create -n prclz_env_pygeos 
source activate prclz_env_pygeos
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install pygeos --channel conda-forge
conda install requests
conda install urlpath
conda install geopandas
```
which will create a cona environment called prclz_env_pygeos, download the packages in the correct order, and install the relevant requirements. 


Alternatively, you could use venv like so:
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

You can also pass in a list of comma-delimited GADM codes to only download some countries, force the command to overwrite any existing files, and show a progress bar:
```
prclz download geofabrik . --countries SLE,LBR --overwrite --verbose
```

If you run this command from whatever your root directory is for prclz, your file structure will look like the following:

```
root
  ├── GADM
  ├── blocks
  ├── buildings
  ├── cache
  ├── complexity
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  └── parcels

```

### GADM 
The same `download` subcommand is used to download national and subnational administative boundaries from the GADM database.
```
prclz download gadm /path/to/output
```

Like the `geofabrik` datasource, the `gadm` datasource supports filtering and overwrite options:
```
prclz download gadm . --countries SLE,LBR --overwrite
```

If you run this command from whatever your root directory is for prclz, your file structure will look like the following:

```
root
  ├── GADM
        ├── LBR
            ...
        ├── SLE 
             ├── license.txt
             ├── gadm36_SLE_3.shx
             ├── gadm36_SLE_3.shp
             ├── gadm36_SLE_3.prj
             ├── gadm36_SLE_3.dbf
             ├── gadm36_SLE_3.cpg
             ├── gadm36_SLE_2.shx
             ├── gadm36_SLE_2.shp
             ├── gadm36_SLE_2.prj
             ├── gadm36_SLE_2.dbf
             ├── gadm36_SLE_2.cpg
             ├── gadm36_SLE_1.shx
             ├── gadm36_SLE_1.shp
             ├── gadm36_SLE_1.prj
             ├── gadm36_SLE_1.dbf
             ├── gadm36_SLE_1.cpg
             ├── gadm36_SLE_0.shx
             ├── gadm36_SLE_0.shp
             ├── gadm36_SLE_0.prj
             ├── gadm36_SLE_0.dbf
             ├── gadm36_SLE_0.cpg
  ├── blocks
  ├── buildings
  ├── cache
  ├── complexity
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  ├── parcels
  └── zipfiles
        ├── gadm36_SLE_shp.zip
        ├── gadm36_LBR_shp.zip

```

## 2. preprocess (optional)

### select building footprints and road networks from Geofabrik layers

The relevant geometry for this workflow resides in the `lines`, `polygons`, and `multipolygons` layers of the Geofabrik downloaded file. The relevant operations to extract these features are expressed most succintly with GDAL file format conversion tools; we provide a wrapper function called `extract` to get these layers:

```
prclz extract osm_file.pbf output_directory/ --overwrite
```

If you were to run this with the sierra-leone-latest.osm.pbf file downloaded in one of the previous steps, your call would look like the following if run from your root directory for prclz:

```
prclz extract geofabrik/sierra-leone-latest.osm.pbf geofabrik/
```

Similar to other functions, you can add an overwrite flag to this command. The command above would result in this file structure:

```
root
  ├── GADM
        ├── LBR
            ...
        ├── SLE 
            ...
  ├── blocks
  ├── buildings
  ├── cache
  ├── complexity
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
        ├── sierra-leone_lines.geojson
        ├── sierra-leone_building_linestrings.geojson
        ├── sierra-leone_building_polygons.geojson
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  ├── parcels
  └── zipfiles
        ├── gadm36_SLE_shp.zip
        ├── gadm36_LBR_shp.zip


```

### split by GADM 

Additionally, the building footprints need to assigned to GADMs in order to be enable parallel processing of `prclz` functions at country-scale. In our current filesystem, we would run split-buildings for Sierra Leone by writing:

```
prclz split-buildings geofabrik/sierra-leone_building_polygons.geojson gadm/SLE/gadm36_SLE_3.shp buildings/
```

After this, our file system will look similar to the following:

```
root
  ├── GADM
        ├── LBR
            ...
        ├── SLE 
            ...
  ├── blocks
  ├── buildings
        ├── buildings_SLE.4.2.1_1.geojson
        ├── buildings_SLE.4.1.4_1.geojson
        ├── buildings_SLE.4.1.3_1.geojson
        ├── buildings_SLE.4.1.2_1.geojson
        ├── buildings_SLE.4.1.1_1.geojson
        ├── buildings_SLE.3.3.8_1.geojson
        ├── buildings_SLE.3.3.4_1.geojson
        ├── buildings_SLE.3.2.5_1.geojson
        ├── buildings_SLE.3.1.8_1.geojson
        ├── buildings_SLE.3.1.13_1.geojson
        ├── buildings_SLE.2.5.7_1.geojson
        ├── buildings_SLE.2.5.5_1.geojson
        ├── buildings_SLE.2.5.1_1.geojson
        ├── buildings_SLE.2.5.11_1.geojson
        ├── buildings_SLE.2.4.9_1.geojson
        ├── buildings_SLE.2.4.5_1.geojson
        ├── buildings_SLE.2.3.6_1.geojson
        ├── buildings_SLE.2.3.5_1.geojson
        ├── buildings_SLE.2.2.3_1.geojson
        ├── buildings_SLE.2.1.7_1.geojson
        ├── buildings_SLE.2.1.1_1.geojson
        ├── buildings_SLE.1.2.4_1.geojson
        ├── buildings_SLE.1.2.16_1.geojson
        ├── buildings_SLE.1.2.13_1.geojson
        ├── buildings_SLE.1.2.12_1.geojson
        └── buildings_SLE.1.1.6_1.geojson
  ├── cache
  ├── complexity
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
        ├── sierra-leone_lines.geojson
        ├── sierra-leone_building_linestrings.geojson
        └── sierra-leone_building_polygons.geojson
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  ├── parcels
  └── zipfiles
        ├── gadm36_SLE_shp.zip
        └── gadm36_LBR_shp.zip
```

## 3. determine city block geometry from road network boundaries 
To determine city block delineations, pass for each administrative unit, pass in the administrative unit boundaries, and the national road network.

```
prclz blocks gadm/SLE/gadm36_SLE_3.shp geofabrik/sierra-leone_lines.geojson blocks/
```

After running this, your file structure should look like the following, with a different date as the name of the log file:

```
root
  ├── GADM
        ├── LBR
            ...
        ├── SLE 
            ...
  ├── blocks
        ├── blocks_gadm36_SLE_3.csv
        ├── logs
              ├── blocks
                    ├── gadm36_SLE_3_2021-07-20T10:43:27.654212.log
  ├── buildings
        ...
  ├── cache
  ├── complexity
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
        ├── sierra-leone_lines.geojson
        ├── sierra-leone_building_linestrings.geojson
        └── sierra-leone_building_polygons.geojson
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  ├── parcels
  └── zipfiles
        ├── gadm36_SLE_shp.zip
        └── gadm36_LBR_shp.zip
```

## 4. calculate block complexity 
To calculate the city block complexity measure, pass in the city block geometry and building footprints. You will need to pass in the building footprint files one at a time. For example:
```
prclz complexity blocks/blocks_gadm36_SLE_3.csv buildings/buildings_SLE.1.1.6_1.geojson complexity/ --overwrite
```
Similar to other functions, you can pass an overwrite flag (--overwrite) to complexity. The call above will result in a file structure similar to the following:

```
root
  ├── GADM
        ├── LBR
            ...
        ├── SLE 
            ...
  ├── blocks
        ├── blocks_gadm36_SLE_3.csv
        ├── logs
              ├── blocks
                    ├── gadm36_SLE_3_2021-07-20T10:43:27.654212.log
  ├── buildings
        ...
  ├── cache
  ├── complexity
        ├── complexity_SLE.1.1.6_1.csv
  ├── errors
  ├── geofabrik
        ├── sierra-leone-latest.osm.pbf
        ├── liberia-latest.osm.pbf
        ├── sierra-leone_lines.geojson
        ├── sierra-leone_building_linestrings.geojson
        └── sierra-leone_building_polygons.geojson
  ├── geojson
  ├── geojson_gadm
  ├── input
  ├── lines
  ├── parcels
  └── zipfiles
        ├── gadm36_SLE_shp.zip
        └── gadm36_LBR_shp.zip
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
