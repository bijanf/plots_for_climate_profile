#!/bin/bash
set -ex
rsync -av --partial --progress 'fallah@cluster.pik-potsdam.de://p/projects/klimafolgenonline/all_data/inputdata/ca/climate/obs/kfo_ens_obs.nc'  ./
rsync -av --partial --progress 'fallah@cluster.pik-potsdam.de://p/projects/klimafolgenonline/all_data/inputdata/ca/climate/ssp585/kfo_ens_ssp585.nc'  ./
rsync -av --partial --progress 'fallah@cluster.pik-potsdam.de://p/projects/klimafolgenonline/all_data/inputdata/ca/climate/ssp126/kfo_ens_ssp126.nc'  ./
rsync -av --partial --progress 'fallah@cluster.pik-potsdam.de://p/projects/klimafolgenonline/all_data/inputdata/ca/climate/historical/kfo_ens_historical.nc'  ./

