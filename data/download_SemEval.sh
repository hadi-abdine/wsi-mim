#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267536&authkey=AAttGd4t9Ul90P0'  && \
unzip -o file.zip && \
rm file.zip
