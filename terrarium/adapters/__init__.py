"""
Adapters connect to a data service for two purposes:

1. Request data from the data server
2. Serialize data to a particular schema.
"""
from terrarium.adapters.aquarium import DataRequester, Serializer
