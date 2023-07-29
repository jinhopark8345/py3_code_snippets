
import orjson


def how_to_load_with_orjson():
    with open(json_path, 'r') as f:
        data = orjson.loads(f.read())
