#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

function docker_id() {
    local ID
    ID="$(docker container ls -qf name="^hugsim-docker$")"

    if [ -z "$ID" ]; then
        ID="$(docker container ls -aqf name="^hugsim-docker$")"
        if [ -n "$ID" ]; then
            echo "[INFO] Starting existing container $ID ..." >&2
            docker start "$ID" >/dev/null
        fi
    fi

    if [ -z "$ID" ]; then
        echo "[INFO] Creating new container ..." >&2
        ID="$(
            docker run -d -i --privileged \
                --gpus all \
                --net host \
                --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                -v /home/robosense/hugsim_workspace:/workspace \
                -v /mnt:/mnt \
                -v /usr/share/bash-completion:/usr/share/bash-completion \
                -v /home/robosense/hugsim_workspace/hugsim_home:${HOME} \
                -e HOME=${HOME} \
                -e USER=${USER} \
                -u $(id -u):$(id -g) \
                -e DISPLAY="${DISPLAY}" \
                -e QT_X11_NO_MITSHM=1 \
                -e GDK_SCALE="${GDK_SCALE:-}" \
                -e GDK_DPI_SCALE="${GDK_DPI_SCALE:-}" \
                -w "/workspace" \
                -h "hugsim-docker" \
               --add-host hugsim-docker:127.0.0.1\
                --ipc "host" \
                --name "hugsim-docker" \
               hugsim_image:latest \
                /bin/bash
        )"
    fi

    echo "$ID"
}


function main() {
    CONTAINER_ID=$(docker_id)

    echo "[INFO] Attaching to container $CONTAINER_ID ..."
    docker exec -it "$CONTAINER_ID" /bin/bash
}

main