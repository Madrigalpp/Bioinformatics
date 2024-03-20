# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device)
    criterion = NegativeLogLikelihood(config['network']).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            risk_pred = model(X)
            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            with torch.no_grad():
                risk_pred = model(X)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:
                    flag += 1
                    if flag >= patience:
                        return best_c_index
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index


import os
import torch
import optuna
import prettytable as pt

# Set global settings
logs_dir = 'logs'
models_dir = os.path.join(logs_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs_dir = 'configs_para'
params = [
    ('kirc1', 'KIRC_fold_1.ini'),
    ('kirc2', 'KIRC_fold_2.ini'),
    ('kirc3', 'KIRC_fold_3.ini'),
    ('kirc4', 'KIRC_fold_4.ini'),
    ('kirc5', 'KIRC_fold_5.ini'),
    #
    # ('lgg1', 'LGG_fold_1.ini'),
    # ('lgg2', 'LGG_fold_2.ini'),
    # ('lgg3', 'LGG_fold_3.ini'),
    # ('lgg4', 'LGG_fold_4.ini'),
    # ('lgg5', 'LGG_fold_5.ini'),
    #
    # ('LUAD1', 'LUAD_fold_1.ini'),
    # ('LUAD2', 'LUAD_fold_2.ini'),
    # ('LUAD3', 'LUAD_fold_3.ini'),
    # ('LUAD4', 'LUAD_fold_4.ini'),
    # ('LUAD5', 'LUAD_fold_5.ini'),
    #
    # ('LUSC1', 'LUSC_fold_1.ini'),
    # ('LUSC2', 'LUSC_fold_2.ini'),
    # ('LUSC3', 'LUSC_fold_3.ini'),
    # ('LUSC4', 'LUSC_fold_4.ini'),
    # ('LUSC5', 'LUSC_fold_5.ini'),
    #
    # ('PAAD1', 'PAAD_fold_1.ini'),
    # ('PAAD2', 'PAAD_fold_2.ini'),
    # ('PAAD3', 'PAAD_fold_3.ini'),
    # ('PAAD4', 'PAAD_fold_4.ini'),
    # ('PAAD5', 'PAAD_fold_5.ini'),
    #
    # ('SARC1', 'SARC_fold_1.ini'),
    # ('SARC2', 'SARC_fold_2.ini'),
    # ('SARC3', 'SARC_fold_3.ini'),
    # ('SARC4', 'SARC_fold_4.ini'),
    # ('SARC5', 'SARC_fold_5.ini'),
    #
    # ('SKCM1', 'SKCM_fold_1.ini'),
    # ('SKCM2', 'SKCM_fold_2.ini'),
    # ('SKCM3', 'SKCM_fold_3.ini'),
    # ('SKCM4', 'SKCM_fold_4.ini'),
    # ('SKCM5', 'SKCM_fold_5.ini'),
    #
    # ('STAD1', 'STAD_fold_1.ini'),
    # ('STAD2', 'STAD_fold_2.ini'),
    # ('STAD3', 'STAD_fold_3.ini'),
    # ('STAD4', 'STAD_fold_4.ini'),
    # ('STAD5', 'STAD_fold_5.ini'),

]
patience = 200


def objective(trial):
    global device, models_dir, configs_dir, patience
    name, ini_file = trial.suggest_categorical('dataset', params)
    logger = create_logger(logs_dir)

    logger.info('Running {}({})...'.format(name, ini_file))
    data = os.path.join(configs_dir, ini_file)
    best_c_index = train(data)

    logger.info("The best valid c-index: {}".format(best_c_index))
    logger.info('')
    return best_c_index


import configparser
from sklearn.model_selection import ParameterGrid
import itertools

if __name__ == '__main__':
    logs_dir = 'logs'
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs_dir = 'configs'

    headers = []
    values = []
    for name, ini_file in params:
        logger.info('Running {}({})...'.format(name, ini_file))

        data = os.path.join(configs_dir, ini_file)

        config = configparser.ConfigParser()
        config.read(data)

        # 定义超参数搜索空间
        param_grid = {
            'epochs': [2000,5000],
            'learning_rate': [3e-3, 1e-3],
            'lr_decay_rate': [1e-4, 3e-4],
            'optimizer': ['Adam', 'SGD'],
            'drop': [0.3, 0.5],
            'norm': [True, False],
            'activation': ['ReLU', 'Sigmoid'],
            'l2_reg': [0, 0.1]
        }

        # 获取超参数组合
        param_combinations = list(ParameterGrid(param_grid))

        # 执行超参数搜索
        best_score = float('-inf')
        best_params = None

        for idx, params in enumerate(param_combinations, start=1):
            section_name = f'section{idx}'  # 创建一个新的部分名称，可以根据需要修改
            if not config.has_section(section_name):
                config.add_section(section_name)

            for key, value in params.items():
                config.set(section_name, str(key), str(value))  # 将键和值都转换为字符串


                best_c_index = train(data)
            score = best_c_index

            # 打印当前参数组合及得分
            print("Parameters:", params)
            print("Score:", score)
            print()

            # 更新最佳参数和得分
            if score > best_score:
                best_score = score
                best_params = params.copy()

        for idx, params in enumerate(param_combinations, start=1):
            section_name = f'section{idx}'  # 创建一个新的部分名称，可以根据需要修改
            if not config.has_section(section_name):
                config.add_section(section_name)

            for key, value in params.items():
                config.set(section_name, str(key), str(value))  # 将键和值都转换为字符串

        with open('your_ini_file.ini', 'w') as configfile:
            config.write(configfile)
        headers.append(name)
        values.append('{:.6f}'.format(score))
        # 打印最佳参数

        print("Best Parameters:", best_params,name)





