#!/usr/bin/env bash
git tag $TAG
git push origin $TAG
poetry publish -r pypi