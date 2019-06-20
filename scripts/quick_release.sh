#!/usr/bin/env bash

ARGVER=""
PUBLISH=0
REPO="pypi"

COLOR="\e[92m"
while [ "$1" != "" ]; do
    case $1 in
        -v | --version )        shift
                                ARGVER=$1
                                ;;
        -p | --publish )        PUBLISH=1
                                ;;
        -r | --repo )           shift
                                REPO=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usages
                                exit 1
    esac
    shift
done



if [ "$ARGVER" != "" ]; then
    cmsg="Releasing version $NAME $ARGVER (publish=$PUBLISH, repo=$REPO) Continue (y/n)?"
else
    current=$(poetry run version)
    cmsg="Bumping version $NAME $current (publish=$PUBLISH, repo=$REP). Continue (y/n)?"
fi

echo -e "$COLOR $cmsg"
read continue
if [ "$continue" != "y" ]; then
    echo -e "Canceled by user"
    exit 0
fi

poetry version $ARGVER
NAME=$(poetry run name)
poetry run upver
VER=$(poetry run version)
TAG=v$VER

if [ "$ARGVER" != "" -a "$VER" != "$ARGVER" ]; then
    echo -e "ERROR: Package version $VER does not match argument version $ARGVER"
    exit 1
fi




STEPS=4

# formatting
echo
echo -e "********************"
echo -e "1/$STEPS FORMATTING"
echo -e "********************"
msg="formatting for release $VER"
echo -e $msg
make format
if [ "$PUBLISH" == 1 ]; then
    git add .
    git commit -m "$msg"
    echo -e "$?"
else
    echo -e "skipping format commit"
fi

# update docs
echo
echo -e "********************"
echo -e "2/$STEPS DOCUMENTATION"
echo -e "********************"
msg="updating docs for release $VER"
echo -e $msg
make docs

if [ "$PUBLISH" == 1 ]; then
    git add .
    git commit -m "$msg"
else
    echo -e "skipping document commit"
fi

# tagging
echo
echo -e "********************"
echo -e "3/$STEPS DOCUMENTATION"
echo -e "********************"
msg="tagging $TAG"
echo -e $msg


if [ "$PUBLISH" == 1 ]; then
    git tag $TAG
else
    echo -e "skipping tagging"
fi

echo
echo -e "$COLOR Push changes to github (y/n)?"
read continue
if [ "$continue" == "y" ]; then
    git push
    git push $TAG
fi

# releasing
echo
echo -e "********************"
echo -e "4/$STEPS Publishing"
echo -e "********************"

if [ "$REPO" != "" -a "$PUBLISH" == 1 ]; then
    poetry publish
else
    echo -e "Skipping publishing"
fi