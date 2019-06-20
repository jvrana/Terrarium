#!/usr/bin/env bash

# get version info
poetry version $1
NAME=$(poetry run name)
VER=$(poetry run version)
TAG=v$VER
poetry run upver

# formatting
msg="formatting for release $VER"
echo $msg
make format
git add .
git commit -m $msg

# update docs
msg="updating docs for release $VER"
echo $msg
make docs
git add .
git commit -m $msg

# tagging
msg="tagging $TAG"
echo $msg
git tag $TAG
git push origin $TAG

# releasing
poetry publish -r pypi