## How to split large building .shp files into GADM-level .geojson files

Iterate through Zipped .shp building files and split them into GADM-level .geojson files for each country.

## Data source

#### Download Africa data to local machine: 
https://platform.ecopiatech.com/data/index

#### Transfer data to midway
```
scp -r /users/path/downloads <cnetid>@midway.rcc.uchicago.edu:/project2/bettencourt/mnp/analytics/data/ecopia/buildings
```

## Command Line Arguments
    
#### Argument definitions:

1. *codes_file*: path to .csv listing all the downloaded building files and associated 3-character country code. `codes_file=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/ecopia_country_codes.csv`

| file_name  | country_code |
| ------------- | ------------- |
| africa_angola_building_32735.dl.zip | AGO |
| africa_benin_building_32631.dl.zip | BEN |
| africa_botswana_building_32734.dl.zip | BWA |

2. *log_path*: path to .log file (automatically created). `log_path=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/ecopia_building_split.log`

3. *progress_file*: path to .csv record of processed building files with the below format (automatically created). `progress_file=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/finished_files.csv`


| file_name	| country_code | outcome |
| ------------- | ------------- | ------------- |
| africa_angola_building_32735.dl.zip | AGO | completed |
| africa_benin_building_32631.dl.zip | BEN | completed |
| africa_burkina_faso_building_32630.dl.zip | BFA | completed |

4. *gadm_path*: path to folder containing downloaded .shp GADM files. `gadm_path=/project2/bettencourt/mnp/analytics/data/gadm`

5. *input_dir*: path to folder containing downloaded .shp building files. `input_dir=/project2/bettencourt/mnp/analytics/data/ecopia/buildings`

6. *output_dir*: path to folder containing split .geojson building files. `output_dir=/project2/bettencourt/mnp/analytics/data/buildings`

#### Passing arguments to script:

```
python split_buildings.py --log_path $log_path --codes_file $codes_file --progress_file $progress_file --gadm_path $gadm_path --input_dir $input_dir --output_dir $output_dir
```

## Midway environment

`ssh <cnetid>@midway2.rcc.uchicago.edu`

`mkdir /project2/bettencourt/mnp/analytics/scripts`

### Build environment
```
module load python/anaconda-2020.02
conda create -n splitter_pygeos 
source activate splitter_pygeos
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install pygeos --channel conda-forge
conda install requests
conda install urlpath
conda install geopandas
```

### Deploy job on Midway

#### `.sbatch` template

```
#!/bin/bash

#SBATCH --job-name=splitter-script
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=56000
#SBATCH --output=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/split_job.out
#SBATCH --error=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/split_job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmarchio@uchicago.edu
#SBATCH --time=24:00:00
#SBATCH --account=pi-bettencourt

cd /project2/bettencourt/mnp/analytics/scripts/split_ecopia

module load python/anaconda-2020.02
source activate splitter_pygeos 

log_path=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/ecopia_building_split.log
codes_file=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/ecopia_country_codes.csv
progress_file=/project2/bettencourt/mnp/analytics/scripts/split_ecopia/finished_files.csv
gadm_path=/project2/bettencourt/mnp/analytics/data/gadm
input_dir=/project2/bettencourt/mnp/analytics/data/ecopia/buildings
output_dir=/project2/bettencourt/mnp/analytics/data/buildings

python split_buildings.py --log_path $log_path --codes_file $codes_file --progress_file $progress_file --gadm_path $gadm_path --input_dir $input_dir --output_dir $output_dir
```

#### Run `bigmem2` [.sbatch script](https://github.com/mansueto-institute/prclz/blob/master/scripts/split-buildings/midway-deployment/split_buildings_job_midway_big.sbatch) on large files using this building [codes_file](https://github.com/mansueto-institute/prclz/blob/master/scripts/split-buildings/midway-deployment/ecopia_country_codes_big.csv)

```
module load python/anaconda-2020.02
source activate splitter_pygeos
sbatch /project2/bettencourt/mnp/analytics/scripts/split_ecopia/split_buildings_job_midway_big.sbatch
squeue --user=<cnetid>
```

#### Run `broadwl` [.sbatch script](https://github.com/mansueto-institute/prclz/blob/master/scripts/split-buildings/midway-deployment/split_buildings_job_midway.sbatch) on rest of files using this building [codes_file](https://github.com/mansueto-institute/prclz/blob/master/scripts/split-buildings/midway-deployment/ecopia_country_codes.csv)

```
module load python/anaconda-2020.02
source activate splitter_pygeos
sbatch /project2/bettencourt/mnp/analytics/scripts/split_ecopia/split_buildings_job_midway_big.sbatch
squeue --user=<cnetid>
```

## Local dev

#### Build local environment
```
conda create --name splitter_pygeos python=3.9.6
source activate splitter_pygeos
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install pygeos --channel conda-forge
conda install requests
conda install urlpath
conda install geopandas
conda install ipykernel
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install jupyter
python -m ipykernel install --user --name=splitter_pygeos 
```

#### Launch notebook
```
jupyter notebook
jupyter notebook stop
```

#### Sample bash script
```
cd /users/path/to/folder
source splitter_pygeos

log_path=/users/path/to/folder/filelog.log
codes_file=/users/path/to/folder/ecopia-codes_test.csv
progress_file=/users/path/to/folder/ecopia_finished.csv
gadm_path=/users/path/to/folder/gadm
input_dir=/users/path/to/folder/ecopia/buildings
output_dir=/users/path/to/folder/buildings

python split_buildings.py --log_path $log_path --codes_file $codes_file --progress_file $progress_file --gadm_path $gadm_path --input_dir $input_dir --output_dir $output_dir
```

