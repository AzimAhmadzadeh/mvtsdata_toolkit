import CONSTANTS as CONST
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import tqdm
import zipfile
import http.client
from hurry.filesize import size


class DataRetriever:
    """
    This is a simple class to provide data for the user to play around with the package's
    functionalities through its demo notebook.
    notebook.
    """

    def __init__(self):
        self.datasets = {1: CONST.URL_DATASET_01,
                         2: CONST.URL_DATASET_02}
        self.meta: http.client.HTTPMessage = None
        self.all_members = None

    def get_info(self, dataset_number=1):
        """
        :param dataset_number:
        :return:
        """
        with urlopen(self.datasets[dataset_number]) as zipresp:
            self.meta = zipresp.info()

            with ZipFile(BytesIO(zipresp.read())) as zfile:
                self.all_members = zfile.namelist()

    def get_total_size(self):
        """:return the size of the data to be extracted."""
        return size(int(self.meta['Content-Length']))

    def get_compression_type(self):
        """:return the compression type of the dataset."""
        return self.meta['Content-Type']

    def get_total_number_of_files(self):
        """:return the total number of files in the dataset."""
        return len(self.all_members) - 1  # -1 for the parrent directory

    def retrieve(self, target_path, dataset_id=1):
        """
        downloads and extracted the compressed dataset (given by 'dataset_id') into the directory
        'target_path'.
        :param target_path:
        :param dataset_id:
        :return:
        """
        with urlopen(self.datasets[dataset_id]) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                for member in tqdm.tqdm(zfile.infolist(), desc='Extracting'):
                    try:
                        zfile.extract(member, target_path)
                    except zipfile.error as e:
                        pass