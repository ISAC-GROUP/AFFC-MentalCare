import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model import *
from tqdm import tqdm
from utils import Averager
from sklearn.model_selection import KFold
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
from deal_data import *

ROOT = os.getcwd()

class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        self.log_file = "results.txt"
        self.log_file_cross = "results/" + args.model + '_cross_results.txt'
        file = open(self.log_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)num_epochs:" + str(args.max_epoch) +
                   "\n5)batch_size:" + str(args.batch_size) +
                   "\n6)dropout:" + str(args.dropout) +
                   "\n7)hidden_node:" + str(args.hidden) +
                   "\n8)input_shape:" + str(args.input_shape) + '\n')
        file.close()

    def n_fold_CV(self, reproduce=False):

        print('Loading data of per subject ... ')
        src_dataset = MyDataSet(self.args.sub_id, 'train')
        tgt_dataset = MyDataSet(self.args.sub_id, 'test')

        src_dataset_train_size = int(len(src_dataset) * 0.8)  
        src_dataset_valid_size = len(src_dataset) - src_dataset_train_size
        torch.manual_seed(2025)
        src_dataset_train, src_dataset_valid = data.random_split(src_dataset,
                                                                 [src_dataset_train_size, src_dataset_valid_size])
        if self.args.model in ['CDPT','CDPT_wo_cl', 'CDPT_wo_modality_encoder', 'CDPT_wo_con_dis','CDPT_wo_pos_tran',
        'CDPT_wo_ppg_ired_gsr','CDPT_wo_ir_gsr_skt','CDPT_wo_ired_gsr_skt','CDPT_w_ired','CDPT_w_gsr','CDPT_w_ppg', 'CDPT_w_ir',
        'CDPT_w_skt','CDPT_wo_ir_ired_skt','CDPT_wo_ppg','CDPT_wo_ir_ired_gsr','CDPT_wo_ppg_gsr_skt','CDPT_wo_ppg_ired_skt',
        'CDPT_wo_ppg_ir_skt','CDPT_wo_ppg_ir_gsr','CDPT_wo_ppg_ir_ired','CDPT_wo_ired_skt','CDPT_wo_ired_gsr','CDPT_wo_ir_skt',
        'CDPT_wo_ir_gsr','CDPT_wo_ir_ired','CDPT_wo_ppg_ired','CDPT_wo_skt','CDPT_wo_gsr_skt','CDPT_wo_ir','CDPT_wo_ired','CDPT_wo_gsr',
        'CDPT_wo_ppg_gsr','CDPT_wo_ppg_skt','CDPT_wo_ppg_ir']:
            train_dataloader = DataLoader(src_dataset_train, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(src_dataset_valid, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            test_dataloader = DataLoader(tgt_dataset, batch_size=self.args.batch_size, drop_last=True)
        else:
            train_dataloader = DataLoader(src_dataset_train, batch_size=self.args.batch_size, shuffle=True)
            valid_dataloader = DataLoader(src_dataset_valid, batch_size=self.args.batch_size, shuffle=True)
            test_dataloader = DataLoader(tgt_dataset, batch_size=self.args.batch_size)

        print('--------------------------------------')
        _ = train(args=self.args, train_loader=train_dataloader, val_loader=valid_dataloader, subject=self.args.sub_id)
        # test the model on testing data
        _, pred, act = test(args=self.args, test_loader=test_dataloader, reproduce=self.args.reproduce)

        acc, pre, rec, f1, con_mat = get_metrics(y_pred=pred, y_true=act)

        self.log3txt('Final: test Sub:{}, ACC:{} PRE:{} REC:{} F1:{}: con_mat:{}'.format(self.args.sub_id, acc, pre, rec, f1, con_mat))
        results = 'Test Sub:{}, ACC:{} PRE:{} REC:{} F1:{}'.format(self.args.sub_id , acc, pre, rec, f1)
        self.log2txt(results)
        print(results)

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.log_file, 'a')
        file.write(str(content) + '\n')
        file.close()

    def log3txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.log_file_cross, 'a')
        file.write(str(content) + '\n')
        file.close()





