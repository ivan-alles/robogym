#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import json
import os

with open(os.path.join(os.path.dirname(__file__), 'package.json'), 'r', encoding='utf-8') as f:
    package = json.load(f)

__version__ = package["version"]
