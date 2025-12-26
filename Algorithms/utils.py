import os
import time
import pprint
from scipy.signal import welch
# from scipy.integrate import simps
import numpy as np
import mne
from scipy import signal
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from model.MDNet import MDNet
from model.CDPT_wo_cl import CDPT_wo_cl
from model.CDPT_wo_modality_encoder import CDPT_wo_modality_encoder
from model.CDPT_wo_con_dis import CDPT_wo_con_dis
from model.CDPT_wo_pos_tran import CDPT_wo_pos_tran
from model.CDPT_wo_ppg import CDPT_wo_ppg
from model.CDPT_wo_ir import CDPT_wo_ir
from model.CDPT_wo_ired import CDPT_wo_ired
from model.CDPT_wo_gsr import CDPT_wo_gsr
from model.CDPT_wo_skt import CDPT_wo_skt
from model.CDPT_wo_ppg_ir import CDPT_wo_ppg_ir
from model.CDPT_wo_ppg_ired import CDPT_wo_ppg_ired
from model.CDPT_wo_ppg_gsr import CDPT_wo_ppg_gsr
from model.CDPT_wo_ppg_skt import CDPT_wo_ppg_skt
from model.CDPT_wo_ir_ired import CDPT_wo_ir_ired
from model.CDPT_wo_ir_gsr import CDPT_wo_ir_gsr
from model.CDPT_wo_ir_skt import CDPT_wo_ir_skt
from model.CDPT_wo_ired_gsr import CDPT_wo_ired_gsr
from model.CDPT_wo_ired_skt import CDPT_wo_ired_skt
from model.CDPT_wo_gsr_skt import CDPT_wo_gsr_skt
from model.CDPT_wo_ppg_ir_ired import CDPT_wo_ppg_ir_ired
from model.CDPT_wo_ppg_ir_gsr import CDPT_wo_ppg_ir_gsr
from model.CDPT_wo_ppg_ir_skt import CDPT_wo_ppg_ir_skt
from model.CDPT_wo_ppg_ired_gsr import CDPT_wo_ppg_ired_gsr
from model.CDPT_wo_ppg_ired_skt import CDPT_wo_ppg_ired_skt
from model.CDPT_wo_ppg_gsr_skt import CDPT_wo_ppg_gsr_skt
from model.CDPT_wo_ir_ired_gsr import CDPT_wo_ir_ired_gsr
from model.CDPT_wo_ir_ired_skt import CDPT_wo_ir_ired_skt
from model.CDPT_wo_ir_gsr_skt import CDPT_wo_ir_gsr_skt
from model.CDPT_wo_ired_gsr_skt import CDPT_wo_ired_gsr_skt
from model.CDPT_w_ppg import CDPT_w_ppg
from model.CPDT_w_ir import CDPT_w_ir
from model.CDPT_w_ired import CDPT_w_ired
from model.CDPT_w_gsr import CDPT_w_gsr
from model.CDPT_w_skt import CDPT_w_skt
import h5py

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def get_model(args):
    if args.model == 'MDNet':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  # 32
                self.subject_num = 74  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.alpha = 0.003
                self.beta = 0.0002
                self.gamma = 0.02
                self.delta = 0.4
                self.epsilon = 0.05
        config = Config_G(args)
        model = MDNet(config, args.test_mode)

    elif args.model in ['CDPT', 'CDPT_wo_cl', 'CDPT_wo_modality_encoder', 'CDPT_wo_con_dis', 'CDPT_wo_pos_tran']:
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64 
                self.subject_num = 72 
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5        # diff_s_loss
                self.epsilon = 0.1      # recon_s_loss
                self.adaptive = 0.02    # adaptive_loss
                self.zeta = 0.05        # contrastive_s_loss
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        if args.model == 'CDPT':
            model = CDPT(config, args.test_mode)
        elif args.model == 'CDPT_wo_cl':
            model = CDPT_wo_cl(config, args.test_mode)
        elif args.model == 'CDPT_wo_modality_encoder':
            model = CDPT_wo_modality_encoder(config, args.test_mode)
        elif args.model == 'CDPT_wo_con_dis':
            model = CDPT_wo_con_dis(config, args.test_mode)
        elif args.model == 'CDPT_wo_pos_tran':
            model = CDPT_wo_pos_tran(config, args.test_mode)
    

    elif args.model == 'CDPT_wo_ppg_ired_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ired_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_gsr_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_gsr_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ired_gsr_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ired_gsr_skt(config, args.test_mode)

    elif args.model == 'CDPT_w_ired':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_w_ired(config, args.test_mode)

    elif args.model == 'CDPT_w_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_w_gsr(config, args.test_mode)

    elif args.model == 'CDPT_w_ppg':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_w_ppg(config, args.test_mode)

    elif args.model == 'CDPT_w_ir':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_w_ir(config, args.test_mode)

    elif args.model == 'CDPT_w_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_w_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_ired_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_ired_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_ired_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_ired_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_gsr_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_gsr_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ired_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ired_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ir_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ir_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ir_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ir_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ir_ired':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ir_ired(config, args.test_mode)

    elif args.model == 'CDPT_wo_ired_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ired_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ired_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ired_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir_ired':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir_ired(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ired':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ired(config, args.test_mode)

    elif args.model == 'CDPT_wo_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_gsr_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_gsr_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ir':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ir(config, args.test_mode)

    elif args.model == 'CDPT_wo_ired':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ired(config, args.test_mode)

    elif args.model == 'CDPT_wo_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_gsr':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_gsr(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_skt':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_skt(config, args.test_mode)

    elif args.model == 'CDPT_wo_ppg_ir':
        class Config_G:
            def __init__(self, args):
                self.batch_size = args.batch_size  
                self.subject_out_dim = 640
                self.hidden_size = 64  
                self.subject_num = 72  
                self.dropout = 0.3
                self.num_classes = 4  
                self.activation = nn.ReLU
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.delta = 0.5
                self.epsilon = 0.1
                self.adaptive = 0.02
                self.zeta = 0.05
                self.topM = 3
                self.tau = 0.9
        config = Config_G(args)
        model = CDPT_wo_ppg_ir(config, args.test_mode)
    return model

def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, pre, rec, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err

# def generate_TS_channel_order(original_order: list):
#     """
#     This function will generate the channel order for TSception
#     Parameters
#     ----------
#     original_order: list of the channel names

#     Returns
#     -------
#     TS: list of channel names which is for TSception
#     """
#     chan_name, chan_num, chan_final = [], [], []
#     for channel in original_order:
#         chan_name_len = len(channel)
#         k = 0
#         for s in [*channel[:]]:
#             if s.isdigit():
#                 k += 1
#         if k != 0:
#             chan_name.append(channel[:chan_name_len - k])
#             chan_num.append(int(channel[chan_name_len - k:]))
#             chan_final.append(channel)
#     chan_pair = []
#     for ch, id in enumerate(chan_num):
#         if id % 2 == 0:
#             chan_pair.append(chan_name[ch] + str(id - 1))
#         else:
#             chan_pair.append(chan_name[ch] + str(id + 1))
#     chan_no_duplicate = []
#     [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if f not in chan_no_duplicate]
#     return chan_no_duplicate[0::2] + chan_no_duplicate[1::2]




