class SchemaValidationError(Exception):
    """Error for schema validate errors"""

    @classmethod
    def raise_from_errors(cls, msg, errors, prefix=""):
        validation_msg = msg + "\n"
        validation_msg += "\n".join(
            ["{} ({}) - {}".format(prefix, i, e) for i, e in enumerate(errors)]
        )
        raise cls(validation_msg)
