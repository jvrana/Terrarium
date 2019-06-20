import toml
from os.path import dirname, abspath, join
import json
from os import getenv
import sys

here = dirname(abspath(__file__))
INPATH = join(here, "../../pyproject.toml")
OUTPATH = join(here, "version.json")


def pull_version():
    """Pulls version from pyproject.toml and moves to version JSON"""
    config = toml.load(INPATH)
    ver_data = dict(config["tool"]["poetry"])
    try:
        with open(OUTPATH, "r") as f:
            print(">> Previous Version:")
            print(f.read())
    except:
        pass

    with open(OUTPATH, "w") as f:
        print("<< New Version (path={}):".format(OUTPATH))
        print(json.dumps(ver_data, indent=2))
        json.dump(ver_data, f, indent=2)


def parse_version():
    """Parses version JSON"""
    with open(join(OUTPATH), "r") as f:
        ver = json.load(f)
    return ver


if __name__ == "__main__":
    pull_version()


ver = parse_version()


def get_version():
    v = ver["version"]
    print(v)
    return v


def get_name():
    name = ver["name"]
    print(name)
    return name


def verify_ci():
    tag = getenv("CIRCLE_TAG")

    if tag != ver["version"]:
        info = "Git tag: {0} does not match the version of this app: {1}".format(
            tag, ver["version"]
        )
        sys.exit(info)


__version__ = ver["version"]
__title__ = ver["name"]
__author__ = ver["authors"]
__homepage__ = ver["homepage"]
__repo__ = ver["repository"]
