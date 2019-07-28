from pydent import AqSession, login
from terrarium.parser import JSONInterpreter
import fire
import json


class TerrariumCLI(object):
    """
    Usage: terrarium [username] [url] parse [filename]
    """

    def __init__(self):
        self._session = None

    def login(self, username, url, password=None):
        """
        Login to Aquarium and return CLI.

        :param username: Aquarium username
        :param url: Aquarium url
        :return: self
        """
        if password:
            self._session = login(username, url, password)
        else:
            self._session = login(username, url)

        return self

    def design(self, filepath, dry_run=False):
        """
        Parse an input JSON.

        :param filepath: path to input JSON
        :param dry_run: if True, will not submit to Aquarium server.
        :return: self

        """
        interpreter = JSONInterpreter(self._session.with_cache(timeout=60))
        with open(filepath, "r") as f:
            interpreter.parse(json.load(f))
        if not dry_run:
            plan_dict = interpreter.submit()
            for plan_name, plan in plan_dict.items():
                print(
                    "{name} id={id} url={url}".format(
                        name=plan_name, id=plan.plan.id, url=plan.url
                    )
                )


def main():
    fire.Fire(TerrariumCLI)
