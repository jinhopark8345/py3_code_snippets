from pprint import pprint
from typing import Dict

import yaml
from yaml.loader import SafeLoader



def load_yaml_demo(yaml_path: str) -> Dict:
    # Open the file and load the file
    with open(yaml_path, "r", encoding="utf-8") as f:
        # data = yaml.load(f, Loader=SafeLoader)
        data = yaml.load(f, Loader=SafeLoader)
    return data


def dump_yaml_demo(dump_yaml_path: str, data: Dict) -> None:
    # without allow_unicode, korean will break
    yaml_obj = yaml.dump(data, sort_keys=True, allow_unicode=True)
    pprint(yaml_obj)

    with open(dump_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)


def main():
    yaml_path = "resources/yaml_demo.yaml"
    data = load_yaml_demo(yaml_path)
    dump_yaml_demo("resources/yaml_save_output.yaml", data)


if __name__ == "__main__":
    main()
