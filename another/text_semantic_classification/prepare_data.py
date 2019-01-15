import csv
import http
import logging
import os
import re
import subprocess
from typing import List

import requests
import tqdm

log = logging.getLogger(__name__)

SEMANTIC_TEXTS_DATA_PATH = ('http://www.cs.cornell.edu/people/pabo/'
                            'movie-review-data/review_polarity.tar.gz')
TEXTS_SEPARATOR = '\t'
SEMANTIC_TEXTS_FILE = 'semantic.tsv'


def main() -> None:
    log.info('Start prepare data')
    gz_semantic_words = 'review_polarity.tar.gz'

    download(gz_semantic_words)
    extract_data(gz_semantic_words)

    join_texts_to_file('txt_sentoken/neg', is_positive=0)
    join_texts_to_file('txt_sentoken/pos', is_positive=1)

    cleaner(files=[gz_semantic_words], directories=['txt_sentoken'])


def download(to_file: str) -> None:
    log.info('Start downloading to file=%s', to_file)
    response = requests.get(SEMANTIC_TEXTS_DATA_PATH)

    if response.status_code != http.HTTPStatus.OK:
        log.warning('Can not to download response=%s', response)
        raise Exception('Can not download semantic words from %s' % SEMANTIC_TEXTS_DATA_PATH)

    with open(to_file, 'wb') as f:
        f.write(response.content)

    log.info('Complete downloading')


def extract_data(from_file: str) -> None:
    log.info('Start unzipping data')
    subprocess.call(['tar', '-xvzf', from_file])
    log.info('Success unzipping data')


def join_texts_to_file(texts_path: str, is_positive: int) -> None:
    log.info('Join texts to file; texts_path=%s is_positive=%s', texts_path, is_positive)

    with open(SEMANTIC_TEXTS_FILE, 'a', encoding='utf-8') as join_file:
        writer = csv.writer(join_file, delimiter=TEXTS_SEPARATOR)

        for file_name in tqdm.tqdm(os.listdir(texts_path)):
            file_path = os.path.join(texts_path, file_name)
            write_text(writer, file_path, is_positive)
    log.info('Success join texts to file; texts_path=%s is_positive=%s', texts_path, is_positive)


def write_text(writer, text_path: str, is_positive: int) -> None:
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub(r'[\n,%s]' % TEXTS_SEPARATOR, ' ', text)
        writer.writerow([text.replace('\n', ''), is_positive])


def cleaner(files: List[str], directories: List[str]):
    map(os.remove, files)
    map(os.removedirs, directories)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
