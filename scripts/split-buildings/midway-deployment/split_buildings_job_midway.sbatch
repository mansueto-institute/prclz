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
