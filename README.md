# Cleaned and commented code for Million Neighborhoods Project
## Data download and processing

#### 1. Download GADM data
The GADM data provides boundaries which we use to partition the globe into computationally feasible parts
From within data_processing/
```python
from prclz.data_processing import download_gadm
download_gadm.update_gadm_data(data_root = "/path/to/your/data/directory/")
```


#### 2. Download Geofabrik data
We use Geofabrik to get our OpenStreetMap raw data. Download this for all regions via the following command.
From within data_processing/
```python
from prclz.data_processing import fetch_geofabrik_data
fetch_geofabrik_data.update_geofabrik_data(data_root = "/path/to/your/data/directory/")
```

#### 3. Extract buildings and lines from the raw Geofabrik data [SATEJ]
The raw Geofabrik data is split into country-level files. This step creates a single "buildings" and a "lines" file for each country. The files are in "/data/geojson/"

#### 4. Block extraction [SATEJ]
Blocks are defined as regions fully circumscribed by roads or natural boundaries. Blocks are our most granular unit of analysis. This step extracts those blocks.

#### 5. Split the country-specific building files by GADM
Each country-level file is simply too huge for efficient computation. So, use the GADM boundaries to split the buildings files. This also functions as a data validation and QC point because along with the processed output in "/data/" the script will output country-level summaries about the matching of the OSM buildings with the GADM boundaries including list of non-matched buildings and a .png summary of the matching. 
From within data_processing/
```python
from prclz.data_processing import split_geojson
split_geojson.split_buildings(data_root = "/path/to/your/data/directory/")
```
NOTE: the default behavior is to process All the countries but if you want to process only one, then you can add the
3-letter country code contained in "country_codes.csv"
```python
from prclz.data_processing import split_geojson
split_geojson.split_buildings(data_root = "/path/to/your/data/directory/", gadm_name='DJI')
```

#### 6. Block complexity [SATEJ]

#### 7. Parcelization [NICO]


## Reblocking

```python
from prclz.data_processing import setup_paths
from prclz.prclz import i_topology_utils, i_reblock

region = 'Africa'
gadm_code = 'DJI'
gadm = 'DJI.1.1_1'
block_id = 'DJI.1.1_1_1'

# Function to return data paths for example data 
data_paths = setup_paths.get_example_paths()
parcels_df, buildings_df, blocks_df = i_topology_utils.load_reblock_inputs(data_paths, region, gadm_code, gadm)

reblocking = i_reblock.reblock_block_id(parcels_df, buildings_df, blocks_df, block_id)
```

