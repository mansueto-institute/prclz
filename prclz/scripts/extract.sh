#!/bin/bash  

pbf_path="$1"
parallel_dir="${pbf_path/input/geojson}"
default_prefix="${parallel_dir/-latest.osm.pbf}"
output_prefix="${2:-$default_prefix}"

function extract() { 
    output_name="${default_prefix}_${1}.geojson"
    script="$2"

    echo "osm.pbf <- ${pbf_path}"
    echo "geojson -> ${output_name}"

    OSM_CONFIG_FILE=osmconf.ini ogr2ogr -progress -f GeoJSON ${output_name} ${pbf_path} -sql "${script}"
    echo ""
}


extract "lines"                "select * from lines where natural = 'coastline' or highway is not null or waterway is not null"
extract "building_linestrings" "select * from lines where building is not null"
extract "building_polygons"    "select * from multipolygons where building is not null"
