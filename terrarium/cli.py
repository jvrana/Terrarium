from pydent import AqSession, login
from terrarium.jsoninterpreter import JSONInterpreter
import fire
import json


class TerrariumCLI(object):
    def __init__(self):
        self._session = None

    def login(self, username, url):
        """
        Login to Aquarium and return CLI.

        :param username: Aquarium username
        :param url: Aquarium url
        :return: self
        """
        self._session = login(username, url)

        return self

    def parse(self, filepath):
        """
        Parse an input JSON.

        :param filepath: path to input JSON
        :return: self
        """
        interpreter = JSONInterpreter(self._session.with_cache(timeout=60))
        with open(filepath, "r") as f:
            interpreter.parse(json.load(f))
        return interpreter.submit()


def main():
    fire.Fire(TerrariumCLI)
