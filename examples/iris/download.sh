#!/usr/bin/env bash

wget https://archive.ics.uci.edu/static/public/53/iris.zip
unzip iris.zip
rm iris.zip Index iris.names iris.data
mv bezdekIris.data iris.data
