import numpy as np
import plyvel


class FeatureStore(object):

    def __init__(self, path):
        self.db = plyvel.DB(path, create_if_missing=True)

    def _key(self, key):
        return str(key).encode('utf-8')

    def put(self, key, value, dtype=np.float32, wb=None):
        value = value.astype(dtype)
        obj = wb if wb else self.db
        obj.put(self._key(key), value.tobytes())

    def get(self, key, default_value, dtype=np.float32):
        rval = self.db.get(self._key(key))
        if rval is None:
            return default_value
        return np.frombuffer(rval, dtype=dtype)
