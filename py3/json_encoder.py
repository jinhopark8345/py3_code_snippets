
import json
from typing import Union
from pprint import pprint

def export_json(json_obj, json_path):
    encoder = CompactJSONEncoder(indent=4)
    with open(json_path, 'w', encoding='UTF-8') as f:
        f.write(encoder(json_obj))


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines.
    from : https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92
    """

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 180
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 30
    """Maximum number of items in container that might be put on single line."""

    INDENTATION_CHAR = " "

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs.update({"indent": 4})
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def __call__(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return "[" + ", ".join(self.__call__(el) for el in o) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.__call__(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return (
                        "{ "
                        + ", ".join(
                            f"{self.__call__(k)}: {self.__call__(el)}"
                            for k, el in o.items()
                        )
                        + " }"
                    )
                else:
                    self.indentation_level += 1
                    output = [
                        self.indent_str + f"{json.dumps(k)}: {self.__call__(v)}"
                        for k, v in o.items()
                    ]
                    self.indentation_level -= 1
                    return (
                        "{\n"
                        + ",\n".join(output)
                        + "\n"
                        + self.indent_str
                        + "}"
                    )
            else:
                return "{}"
        elif isinstance(
            o, float
        ):  # Use scientific notation for floats, where appropiate
            return format(o, "g")
        elif isinstance(o, str):  # escape newlines
            o = o.replace("\n", "\\n")
            return f'"{o}"'
        else:
            return json.dumps(o)

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.__call__(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: Union[list, tuple, dict]):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(
                isinstance(el, self.CONTAINER_TYPES) for el in o.values()
            )

    @property
    def indent_str(self) -> str:
        return self.INDENTATION_CHAR * (self.indentation_level * self.indent)



with open('resources/json_demo.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
pprint(data)

from json_encoder import CompactJSONEncoder
with open('resources/json_custom_encoder_output.json', 'w', encoding='UTF-8') as f:
    encoder = CompactJSONEncoder(indent=4)
    f.write(encoder(data))
