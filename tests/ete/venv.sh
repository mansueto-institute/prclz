#!/bin/bash 

echo "current directory: $(pwd)"
if [ -n "${VIRTUAL_ENV}" ]; then 
    echo "virtual environment active; deactivating"
    ./venv/bin/deactivate
fi 

if [ -d "./venv" ]; then 
    echo "removing venv directory"
    rm -r ./venv 
fi 

echo "creating new venv"
python3 -mvenv venv 
source ./venv/bin/activate
pip install wheel

echo "installing prclz"
# handle shapely vs pygeos GEOS version mismatch; see https://github.com/Toblerity/Shapely/issues/651
pip install --no-binary shapely shapely
{
pip install . 
} 2>&1 | sed -e "s/^/    /g"

echo "testing prclz"
./venv/bin/prclz