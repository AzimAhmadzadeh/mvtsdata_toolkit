import CONSTANTS as CONST
from io import BytesIO
import os
from urllib.request import urlopen
from urllib.parse import urlparse
from zipfile import ZipFile
import zipfile
import http.client
import tqdm
from hurry.filesize import size
import yaml


class DataRetriever:
    """
    This is a simple class to provide data for users to play around with the package's
    functionalities through its demo Jupyter notebook.
    """
    def __init__(self, dataset_number=1):

        configs: dict = self.__read_config()
        self.dataset_url = configs['DATASETS'][dataset_number]
        self.all_members = None
        self.dataset_name = self.__get_dataset_name()
        self.__meta: http.client.HTTPMessage = self.__get_info()

    @staticmethod
    def __read_config():
        """
        Reads the config file located at CONST.PATH_TO_DATASETS_CONFIG, which contains a list of
        URLs to a few datasets.

        :return: The content of the configuration file as a dictionary.
        """
        path_to_config = os.path.join(CONST.ROOT, CONST.PATH_TO_DATASETS_CONFIG)
        with open(path_to_config) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        return configs

    def __get_dataset_name(self):
        """
        Retrieves the dataset's name from the dataset's URL.
        """
        url_path = urlparse(self.dataset_url).path
        return os.path.basename(url_path)

    def __get_info(self):
        """
        Retrieves some basic info about the dataset before downloading it.
        """
        with urlopen(self.dataset_url) as req:
            meta = req.info()
        return meta

    def print_info(self):
        print('URL:\t\t{}'.format(self.dataset_url))
        print('NAME:\t\t{}'.format(self.dataset_name))
        print('TYPE:\t\t{}'.format(self.get_compression_type()))
        print('SIZE:\t\t{}'.format(self.get_total_size()))

    def get_total_size(self):
        """
        :return: The size of the data to be extracted.
        """
        return size(int(self.__meta['Content-Length']))

    def get_compression_type(self):
        """
        :return: The compression type of the dataset.
        """
        return self.__meta['Content-Type']

    def get_total_number_of_files(self):
        """
        :return: The total number of files in the dataset.
        """
        if self.all_members is None:
            print('[!] this can be called only after the files are downloaded.')
            return 0
        return len(self.all_members) - 1  # -1 for the parent directory

    def retrieve(self, target_path):
        """
        Downloads and extracts the compressed dataset (given by 'dataset_id') into the directory
        'target_path'. The original compressed file won't be kept. Only it's content will be copied
        into 'target_path'.

        :param target_path: Where the dataset should be copied to.
        """
        with urlopen(self.dataset_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                self.all_members = zfile.namelist()
                for member in tqdm.tqdm(zfile.infolist(), desc='Extracting'):
                    try:
                        zfile.extract(member, target_path)
                    except zipfile.error as e:
                        print(e)

    def test(self):
        with urlopen(self.dataset_url) as zipresp:
            self.__meta = zipresp.info()


def main():
    dr = DataRetriever()
    print('URL:\t\t{}'.format(dr.dataset_url))
    print('NAME:\t\t{}'.format(dr.dataset_name))
    print('TYPE:\t\t{}'.format(dr.get_compression_type()))
    print('SIZE:\t\t{}'.format(dr.get_total_size()))

    print('Now let\'s download the file ...')
    dr.retrieve(target_path='./xxx')
    print('\n\n\t\tDONE')


if __name__ == "__main__":
    main()
