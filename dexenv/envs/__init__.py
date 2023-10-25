from pathlib import Path

from dexenv.utils.common import list_class_names

cur_path = LIB_PATH = Path(__file__).resolve().parent
env_name_to_path = list_class_names(cur_path)
