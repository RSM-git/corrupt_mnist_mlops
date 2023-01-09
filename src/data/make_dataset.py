# -*- coding: utf-8 -*-
import logging
from glob import glob
from pathlib import Path

import click
import numpy as np
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train_sets = glob(f"{input_filepath}/train*.npz")
    train = None
    for train_set in train_sets:
        with np.load(train_set) as data:
            if train is None:
                train = dict(data.items())
            else:
                train["images"] = np.concatenate((train["images"], data["images"]))
                train["labels"] = np.concatenate((train["labels"], data["labels"]))

    with np.load(f"{input_filepath}/test.npz") as data:
        test = dict(data.items())

    train["images"] = torch.tensor(train["images"], dtype=torch.float32)
    train["labels"] = torch.tensor(train["labels"], dtype=torch.long)
    test["images"] = torch.tensor(test["images"], dtype=torch.float32)
    test["labels"] = torch.tensor(test["labels"], dtype=torch.long)

    train["images"] = torchvision.transforms.Normalize(train["images"].mean(), train["images"].std())(train["images"])
    test["images"] = torchvision.transforms.Normalize(train["images"].mean(), train["images"].std())(test["images"])

    torch.save(train, f"{output_filepath}/train.pt")
    torch.save(test, f"{output_filepath}/test.pt")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
