#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267537&authkey=AABusK7vLbNtmjA'  && \
unzip -o file.zip && \
rm file.zip
pip install -U -e ./fairseq/
