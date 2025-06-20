#!/bin/bash

docker build \
  --build-arg http_proxy=http://172.17.0.1:9981 \
  --build-arg https_proxy=http://172.17.0.1:9981 \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) \
  --build-arg HOST_USER=$(id -un) \
  --build-arg HOST_GROUP=$(id -gn) \
  -t hugsim_image .
