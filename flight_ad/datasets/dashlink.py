import urllib.request
import zipfile
from pathlib import Path
from pandas import read_parquet
from tqdm import tqdm
from ._interface import retrieve_json

MODULE_PATH = Path(__file__).parent
DATASET_INFO = retrieve_json(MODULE_PATH / "data.json")
DASHLINK_FILES = MODULE_PATH / "dashlink_flights"


class ProgressBar(tqdm):

    """Download progress bar based on tqdm."""

    def update_to(self, block=1, block_size=1, total_size=None):
        """Update progress bar."""
        if total_size is not None:
            self.total = total_size
        return self.update(block * block_size - self.n)


class DatasetException(Exception):

    """Database exception in case the dataset is not found."""
    pass


def init_folder(folder_path):
    """Create a folder in case it doesn't exist."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    return folder_path


def download_dataset():
    """Download dashlink's dataset."""
    url = DATASET_INFO['dashlink']['url']
    file_name = url.split('/')[-1]
    target_path = MODULE_PATH / file_name
    target_folder = init_folder(DASHLINK_FILES)

    print(f"Downloading {file_name}...")
    with ProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=file_name) as t:
        urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to, data=None)
        t.total = t.n

    print(f"Extracting {file_name}...")
    with zipfile.ZipFile(target_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    target_path.unlink()

    return target_folder


def check_dataset_existence(dataset_path: Path):
    """Check if every file in the dataset exists."""
    present_files = set([f.name for f in dataset_path.glob("*.parquet")])
    required_files = set(DATASET_INFO['dashlink']['files'])
    if present_files == required_files:
        exist = True
    else:
        exist = False

    return exist


def load(f):
    """Default bind function for sample dataset."""
    df = read_parquet(f)
    df['flight_id'] = str(Path(f).stem)

    return df.copy()


def load_dashlink_bindings(data_path=None, download=False):
    """Load data bindings regarding the sample flight dataset."""
    if data_path is None:
        data_path = DASHLINK_FILES
        if not check_dataset_existence(data_path) and download:
            data_path = download_dataset()
        elif not check_dataset_existence(data_path):
            raise DatasetException("Dataset not found.")
    else:
        pass

    bind_function = load

    dashlink_bindings = {}
    for f in data_path.glob("*.parquet"):
        dashlink_bindings[f.stem] = {
            'bind_function': bind_function,
            'data': str(f)
        }

    return dashlink_bindings
