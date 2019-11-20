from validator import Each
from validator import InstanceOf
from validator import Length
from validator import Required
from validator import SubclassOf
from validator import validate
from validator import Validator

_ = Each, Required, Length, validate


class Any(Validator):
    def __init__(self, *values):
        self.values = values

        err_msg = " OR ".join(
            "({}) {}".format(i, v.err_message) for i, v in enumerate(self.values)
        )
        not_msg = " AND ".join(
            "({}) {}".format(i, v.not_message) for i, v in enumerate(self.values)
        )
        self.err_message = "must pass at least one of validations for {}".format(
            self.__class__.__name__, err_msg
        )
        self.not_message = "must fail all validations for {}: {}".format(
            self.__class__.__name__, not_msg
        )

    def __call__(self, value):
        for v in self.values:
            if v(value):
                return True
        return False


class All(Validator):
    def __init__(self, *values):
        self.values = values

        err_msg = " AND ".join(
            "({}) {}".format(i, v.err_message) for i, v in enumerate(self.values)
        )
        not_msg = " OR ".join(
            "({}) {}".format(i, v.not_message) for i, v in enumerate(self.values)
        )
        self.err_message = "must pass all of validations for {}: {}".format(
            self.__class__.__name__, err_msg
        )
        self.not_message = "must fail at least one of validations for {}: {}".format(
            self.__class__.__name__, not_msg
        )

    def __call__(self, value):
        return all([v(value) for v in self.values])


class AnyInstanceOf(Any):
    def __init__(self, *types):
        super().__init__(*[InstanceOf(t) for t in types])


class AnySubclassOf(Any):
    def __init__(self, *types):
        super().__init__(*[SubclassOf(t) for t in types])
