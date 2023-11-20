import logging
import sys

import pandas as pd


def download_data(url: str, path: str):
    data = pd.read_csv(url, sep=';')

    logger = logging.getLogger(__name__)
    logger.info(f"Data downloaded from {url}")

    data.to_csv(path, index=False)


if __name__ == "__main__":
    download_data(sys.argv[1], sys.argv[2])
