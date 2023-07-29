
import orjson

with open(json_path, 'r') as f:
    data = orjson.loads(f.read())
