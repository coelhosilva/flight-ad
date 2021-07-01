from pathlib import Path
from pandas import read_parquet
from ._interface import retrieve_json


MODULE_PATH = Path(__file__).parent
DATASET_INFO = retrieve_json(MODULE_PATH/"data.json")


class DatasetException(Exception):
    pass


def download_dataset():
    return Path(None)


def check_dataset_existence(dataset_path: Path):
    return False


def load(f):
    df = read_parquet(f)
    df['flight_id'] = str(Path(f).stem)

    return df.copy()


def load_dashlink_bindings(data_path=None, download=False):
    if data_path is None:
        data_path = MODULE_PATH/"flights"

    if not check_dataset_existence(data_path) and download:
        data_path = download_dataset()
    elif not check_dataset_existence(data_path):
        raise DatasetException("Dataset not found.")

    bind_function = load

    dashlink_bindings = {}
    for f in data_path.glob("*.parquet"):
        dashlink_bindings[f.stem] = {
            'bind_function': bind_function,
            'data': str(f)
        }

    return dashlink_bindings
