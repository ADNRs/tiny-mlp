#!/usr/bin/env bash

wget https://archive.ics.uci.edu/static/public/94/spambase.zip
unzip spambase.zip
rm -rf spambase.zip spambase.names spambase.DOCUMENTATION
