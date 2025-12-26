import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class MyDataSet(Dataset):
    def __init__(self, test_person, train_or_test):
        data0, data1, data2, data3, data4, domain_label_sum, label_sum = [], [], [], [], [], [], []
        if train_or_test == 'train':
            for i in range(74):
                if i != int(test_person):
                    if i > int(test_person):
                        sub_data0, sub_data1, sub_data2, sub_data3, sub_data4, sub_domain_label, sub_label = self.read_per_subject(i, i-1)
                    else:
                        sub_data0, sub_data1, sub_data2, sub_data3, sub_data4, sub_domain_label, sub_label = self.read_per_subject(i, i)
                    data0.append(sub_data0)
                    data1.append(sub_data1)
                    data2.append(sub_data2)
                    data3.append(sub_data3)
                    data4.append(sub_data4)
                    domain_label_sum.append(sub_domain_label)
                    label_sum.append(sub_label)
            data0 = np.concatenate(data0, axis=0)
            data1 = np.concatenate(data1, axis=0)
            data2 = np.concatenate(data2, axis=0)
            data3 = np.concatenate(data3, axis=0)
            data4 = np.concatenate(data4, axis=0)
            domain_label_sum = np.concatenate(domain_label_sum, axis=0)
            label_sum = np.concatenate(label_sum, axis=0)
            print(data0.shape, data1.shape, domain_label_sum.shape, label_sum.shape)
        else:
            data0, data1, data2, data3, data4, domain_label_sum, label_sum = self.read_per_subject(test_person, 72)
        self.data0 = torch.tensor(data0, dtype=torch.float32)
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.data3 = torch.tensor(data3, dtype=torch.float32)
        self.data4 = torch.tensor(data4, dtype=torch.float32)
        self.label = torch.tensor(label_sum, dtype=torch.long)
        self.domain_label = torch.tensor(domain_label_sum, dtype=torch.long)

    def __getitem__(self, item):
        return self.data0[item], self.data1[item], self.data2[item], self.data3[item], self.data4[item], self.label[item], self.domain_label[item]

    def __len__(self):
        return self.data0.shape[0]

    def norm(self, x):
        return (2 * (x - np.min(x))) / (np.max(x) - np.min(x)) - 1

    def read_per_subject(self, sub, domain_label_value):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        data_path = '/PERCOM26/data/sub' + str(sub) + '.hdf'
        dataset = h5py.File(data_path, 'r')
        data0 = np.array(dataset['data0'])
        data1 = np.array(dataset['data1'])
        data2 = np.array(dataset['data2'])
        data3 = np.array(dataset['data3'])
        data4 = np.array(dataset['data4'])
        label = np.array(dataset['label'])
        domain_label = np.full_like(label, domain_label_value)
        # print('Data:{}'.format(data0))
        print('>>>Sub:{} Data0:{} Data1:{} Data2:{} Data3:{} Data4:{} Domain_label:{} Label:{}'.format(sub, data0.shape, data1.shape, data2.shape,
                                                                                 data3.shape, data4.shape, domain_label.shape, label.shape))
        return data0, data1, data2, data3, data4, domain_label, label
