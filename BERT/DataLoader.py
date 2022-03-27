import glob

import pandas as pd


def get_file_list(directory, extension):
    return glob.glob(directory + '[!~]*.' + extension)


def load_from_xls_file(file):
    pass


class Dataset:

    def __init__(self):
        pass


class TwitterDataset(Dataset):

    def __init__(self, directory):
        self.file_list = get_file_list(directory, 'csv')
        self.current_texts = None
        self.file_num = 0
        self.text_num = 0

    def __iter__(self):
        self.file_num = 0
        self.text_num = 0
        self.open_next_file()
        return self

    def __next__(self):
        text = None
        while not text:
            text = self.next_text()
            if text != text:
                continue
            return text

    def open_next_file(self):
        if self.file_num >= len(self.file_list):
            raise StopIteration

        file = self.file_list[self.file_num]
        print('\nFile', self.file_num + 1, 'of', len(self.file_list))
        print(file)
        csv = pd.read_csv(file, sep='^([^,]+),', engine='python', header=None)
        # Get relevant columns
        self.current_texts = csv[2].to_list()

        self.file_num += 1
        self.text_num = 0

    def next_text(self):
        if self.text_num >= len(self.current_texts):
            self.open_next_file()

        print('Text', self.text_num + 1, 'of', len(self.current_texts), end='\r')
        text = self.current_texts[self.text_num]
        self.text_num += 1
        return text
