# ------------------------------------------------------------------------------
# --coding='utf-8'--

# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import KFold

import os
import torch
import torch.optim as optim
import prettytable as pt

from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger


def train(ini_file, n_splits=5, patience=5):
    ''' Performs training according to .ini file and five-fold cross-validation.

    :param ini_file: (String) the path of .ini file
    :param n_splits: (int) number of folds for cross-validation
    :param patience: (int) patience for early stopping
    :return best_c_index: the best c-index across all folds
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network'])
    criterion = NegativeLogLikelihood(config['network'])
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])

    # constructs data loader based on configuration
    dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_c_index = 0

    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_index),
            batch_size=len(train_index)
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_index),
            batch_size=len(test_index)
        )

        # ... Rest of your training code ...

        # Training loop for the current fold
        for epoch in range(1, config['train']['epochs'] + 1):
    # ... Rest of your training code ...

    return best_c_index
