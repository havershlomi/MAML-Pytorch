import copy
from logging import root
import torch
import os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random
import sys
import pickle
import argparse
import io
from copy import deepcopy

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    args_dict = vars(args).copy()
    del args_dict['root']
    run_name = '-'.join([f'{k}-{args_dict[k]}' for k in args_dict])
    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('mps')

    maml = Meta(args, config).to(device)
    # model_path = './maml_2_epoch-60000-n_way-5-k_spt-1-k_qry-15-imgsz-84-imgc-3-task_num-4-meta_lr-0.001-update_lr-0.01-update_step-5-update_step_test-10.pt'
    # maml = Meta(args, config).to(device)
    # maml.load_state_dict(torch.load(model_path))
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    print('Total trainable tensors:', num)
    rootPath = args.root

    # batchsz here means total episode number
    mini = MiniImagenet(rootPath, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    # mini_val = MiniImagenet(rootPath, mode='val', n_way=args.n_way, k_shot=args.k_spt,
    #                         k_query=args.k_qry,
    #                         batchsz=100, resize=args.imgsz)

    mini_test = MiniImagenet(rootPath, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True,
                        num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(
                device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)
            # TODO: Need to unserstand what step means

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation
                db_val = DataLoader(
                    mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_val:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                        x_qry.squeeze(0).to(
                            device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Eval acc:', accs)
        # need to add evalauation for test set
        model_path = f'./maml_{epoch}_{run_name}.pt'
        # torch.save(maml, model_path)
        torch.save(maml.state_dict(), model_path)

        print(f'model was saved: {model_path}')

    # Test
    test_accs = test(mini_test, maml=maml, device=device)
    print('Test acc:', test_accs)
    # test_accs = test(mini_test, maml=maml, device=device)
    # print('Test acc 2:', test_accs)


def test(mini, maml, device):
    db_test = DataLoader(mini, 1, shuffle=False,
                         num_workers=1, pin_memory=True)
    # should i call maml.finetunning or maml() for testing, probebly the one that doesn't update any params
    accs_all_test = []
    maml = deepcopy(maml)
    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
            x_qry.squeeze(0).to(
                device), y_qry.squeeze(0).to(device)

        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    return accs


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--root', type=str, help='base folder of dataset',
                           default='/Users/shaver/git_tree/project/dataset')
    argparser.add_argument('--epoch', type=int,
                           help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int,
                           help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int,
                           help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int,
                           help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float,
                           help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float,
                           help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int,
                           help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int,
                           help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
