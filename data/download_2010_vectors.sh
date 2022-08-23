#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267515&authkey=APmFp5ghXA7B8Q8'  && \
unzip -o file.zip && \
rm file.zip
