#!/bin/bash

# download IMDb for JOB-light
curl -L -o imdb.tgz http://homepages.cwi.nl/~boncz/job/imdb.tgz
mkdir -p ./IMDB/data/
tar zxvf imdb.tgz -C ./IMDB/data/
rm imdb.tgz