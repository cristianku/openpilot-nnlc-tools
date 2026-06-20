#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_USER="cristianku"
IMAGE_NAME="openpilot-nnlc-tools"
COMPOSE_FILE="docker/docker-compose.yml"

TAG="${1:-latest}"

echo "Building Docker image with docker compose..."
docker compose -f "$COMPOSE_FILE" build

echo "Finding built image ID..."
IMAGE_ID=$(docker compose -f "$COMPOSE_FILE" images -q | head -n 1)

if [ -z "$IMAGE_ID" ]; then
  echo "ERROR: No image found after build."
  exit 1
fi

REMOTE_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"

echo "Tagging image:"
echo "$IMAGE_ID -> $REMOTE_IMAGE"
docker tag "$IMAGE_ID" "$REMOTE_IMAGE"

echo "Pushing to Docker Hub..."
docker push "$REMOTE_IMAGE"

echo "Done:"
echo "$REMOTE_IMAGE"
