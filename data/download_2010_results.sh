#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267516&authkey=APWeFg4qVmhpd_c'  && \
unzip -o file.zip && \
rm file.zip
