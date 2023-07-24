
from detectron2.config import LazyConfig
cfg = LazyConfig.load("./tmp_config2.py")  # an omegaconf dictionary
breakpoint()
assert cfg.a.z.xx == 1
