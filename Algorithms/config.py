import argparse


def set_config():
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--test-mode', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='Mental')
    parser.add_argument('--data-path', type=str, default='/data/data_preprocessed_python')
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    # parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V'])
    # parser.add_argument('--segment', type=int, default=1, help='segment length in seconds')
    parser.add_argument('--batch-size', type=int, default=256)
    # parser.add_argument('--input-shape', type=tuple, default=(1, 32, 512))
    parser.add_argument('--sid', type=int, nargs='+', default=[0, 1, 2, 3, 4, 31], help='List of subject IDs')
    parser.add_argument('--sub-id', type=str, default='56', help='List of subject IDs')

    parser.add_argument('--window', type=str, default='1_1')
    parser.add_argument('--window_sec', type=int, default=2)
    parser.add_argument('-hop_length', default=128, type=int, help='The step size or stride to move the window.')
    parser.add_argument('--overlap', type=float, default=0)

    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--test', type=str, default='LOSO-CV',
                        choices=['LOSO-CV', 'KFOLD-CV']) 
    parser.add_argument('--model', type=str, default='TSception')  
    parser.add_argument('--multi-loss', type=int, default=1, help='Is model has multiple loss function?, EEGMDNet choose false')
    parser.add_argument('--random-seed', type=int, default=2025)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    parser.add_argument('--early-stop', default=True, type=bool, help='Whether need early_stopping method or not.')
    parser.add_argument('--patience', default=35, type=int, help='Tolerate epoch of early_stopping method.')
    parser.add_argument('--prepare-data', type=int, default=0)
    parser.add_argument('--domain-model', type=list, default=['CDPT','CDPT_wo_cl', 'CDPT_wo_modality_encoder', 'CDPT_wo_con_dis','CDPT_wo_pos_tran',
        'CDPT_wo_ppg_ired_gsr','CDPT_wo_ir_gsr_skt','CDPT_wo_ired_gsr_skt','CDPT_w_ired','CDPT_w_gsr','CDPT_w_ppg', 'CDPT_w_ir',
        'CDPT_w_skt','CDPT_wo_ir_ired_skt','CDPT_wo_ppg','CDPT_wo_ir_ired_gsr','CDPT_wo_ppg_gsr_skt','CDPT_wo_ppg_ired_skt',
        'CDPT_wo_ppg_ir_skt','CDPT_wo_ppg_ir_gsr','CDPT_wo_ppg_ir_ired','CDPT_wo_ired_skt','CDPT_wo_ired_gsr','CDPT_wo_ir_skt',
        'CDPT_wo_ir_gsr','CDPT_wo_ir_ired','CDPT_wo_ppg_ired','CDPT_wo_skt','CDPT_wo_gsr_skt','CDPT_wo_ir','CDPT_wo_ired','CDPT_wo_gsr',
        'CDPT_wo_ppg_gsr','CDPT_wo_ppg_skt','CDPT_wo_ppg_ir'])
        
    ######## SPECIAL MODEL SETTING ########
    # LGGNet & TSception
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--save-path', default='./Output/')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--load-path', default='./Output/')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--reproduce', type=bool, default=False)
    args = parser.parse_args()
    return args
