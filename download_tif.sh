#!/bin/bash
set -ex
mkdir -p data
rsync -av --ignore-existing --progress 'fallah@cluster.pik-potsdam.de:/p/projects/gvca/bijan/chelsa_cmip6/downloaded_files/cropped_*'  ./data



