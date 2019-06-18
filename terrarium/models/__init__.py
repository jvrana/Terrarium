# TODO: change data to first class objects for the following reasons: (1) - option to change output json format more easily, (2) clear and consistent model relationship
# TODO: look at schemas for list of necessary attributes


class BaseModel(object):
    pass


class Operation(BaseModel):
    pass


class OperationType(BaseModel):
    pass


class IOType(BaseModel):
    pass


class IOFilter(BaseModel):
    pass


class IOValue(BaseModel):
    pass


class Material(BaseModel):
    """A physical material / inventory that can be consumed or produced by an operation"""

    pass


class MaterialType(BaseModel):
    """Type of material"""

    pass


class MaterialProperties(BaseModel):
    """Material Properties"""


class Collection(BaseModel):
    """A collection of materials"""
