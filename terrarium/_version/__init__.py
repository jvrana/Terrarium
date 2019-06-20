import json
from os.path import dirname, abspath, join
from os import getenv
import sys
from configparser import ConfigParser

VERSION_JSON_PATH = "version.json"
VERSION_TOML = "pyproject.toml"
VERSION_KEY = "tool.poetry"
VERSION_DIR = dirname(abspath(__file__))


def pull_version():
    """Pulls version from pyproject.toml and moves to version JSON"""
    rel_dir = join(VERSION_DIR, "..", "..")
    config_filename = join(rel_dir, VERSION_TOML)

    config = ConfigParser()
    config.read(config_filename)
    print(config)
    ver_data = dict(config[VERSION_KEY])

    clean = lambda s: s.replace('"', "").replace('"', "")

    for k, v in ver_data.items():
        ver_data[k] = clean(v)

    target_path = join(VERSION_DIR, VERSION_JSON_PATH)
    try:
        with open(target_path, "r") as f:
            print(">> Previous Version:")
            print(f.read())
    except:
        pass

    with open(target_path, "w") as f:
        print("<< New Version (path={}):".format(target_path))
        print(json.dumps(ver_data, indent=2))
        json.dump(ver_data, f, indent=2)


def parse_version():
    """Parses version JSON"""
    with open(join(VERSION_DIR, VERSION_JSON_PATH), "r") as f:
        ver = json.load(f)
    return ver


if __name__ == "__main__":
    pull_version()


ver = parse_version()


def get_version():
    v = ver["version"]
    print(v)
    return v


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
