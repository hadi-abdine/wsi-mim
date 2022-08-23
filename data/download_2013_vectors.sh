#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267508&authkey=AEQeBzdjF14nDmI'  && \
unzip -o file.zip && \
rm file.zip
