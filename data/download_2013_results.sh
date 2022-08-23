#!/bin/sh

curl -L -o file.zip 'https://onedrive.live.com/download?cid=AE69638675180117&resid=AE69638675180117%21267517&authkey=AIeuUgV6QIV7IYo'  && \
unzip -o file.zip && \
rm file.zip
