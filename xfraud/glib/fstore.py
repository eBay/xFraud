# Copyright 2020-2021 eBay Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
