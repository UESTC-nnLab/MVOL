from __future__ import print_function
import argparse
import os

import sys

from scipy import misc
import numpy as np
# import sklearn as sk
import sklearn.metrics as sk

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--name', required=True, type=str,
                    help='neural network name and training set')
parser.add_argument('--method', default='msp', type=str, help='ood detection method')
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
parser.add_argument('--epsilon', default=8, type=int, help='epsilon')

parser.add_argument('--if-aux', default=0, type=int, help='whether use auxiliary dataset')
parser.add_argument('--auxiliary-dataset', default='RandomImages300k', required=False,
                    choices=['mixed_cifar10','subset_cifar100','80m_tiny_images', 'imagenet', 'mixed_cifar100','imagenet32', 'RandomImages300k'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--exp-cat-name', required=True, type=str, help='the name of experience categories, i.e. vanilla oe')
parser.add_argument('--model-arch', default='densenet', type=str, required=True, help='model architecture')
parser.set_defaults(argument=True)

args = parser.parse_args()

np.random.seed(1)

def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    # AUPR
    mtype = 'AUPR'
    aupr = get_aupr_measures(known, novel)
    results[mtype]=aupr
    return results

def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1
    print("threshold:{}".format(threshold))
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

def get_aupr_measures(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    aupr = sk.average_precision_score(labels, examples)

    return aupr

def write_excel(results, in_dataset, out_dataset, name, method):
    import pandas as pd
    file_path = 'OOD.xlsx'

    sheet_name = out_dataset
    try:
        df = pd.read_excel(file_path, engine='openpyxl',sheet_name=sheet_name)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Method', 'Out Dataset', 'In Dataset', 'FPR', 'AUROC', 'AUPR'])
        df.to_excel(file_path, sheet_name=sheet_name,index=False)
        df = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)

    start_row = df.shape[0]

    df.loc[start_row] = [name, method, out_dataset, in_dataset, results['FPR'] * 100, results['AUROC'] * 100, results['AUPR'] * 100]

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

def write_id_excel(results, in_dataset, name, method):
    import pandas as pd
    file_path = 'OOD.xlsx'
    sheet_name = 'id'
    df = pd.read_excel(file_path, engine='openpyxl',sheet_name=sheet_name)

    start_row = df.shape[0]

    df.loc[start_row] = [name, method, in_dataset, results['ACC']]

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

def create_excel():
    import pandas as pd

    out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365', 'SVHN']
    out_datasets.append('All')
    file_path = 'OOD.xlsx'
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for out_dataset in out_datasets:
            pd.DataFrame(columns=['Name', 'Method', 'Out Dataset', 'In Dataset', 'FPR', 'AUROC', 'AUPR']).to_excel(writer, sheet_name = out_dataset, index = False)
        pd.DataFrame(columns=['Name', 'Method', 'In Dataset', 'ACC']).to_excel(writer, sheet_name = 'id', index = False)

def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT', 'AUPR']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUPR']), end='')
    print('')


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT', 'AUPR']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

def compute_traditional_ood(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, out_datasets, method, name):
    print('Natural OOD')
    print('nat_in vs. nat_out')

    known = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat','in_scores.txt'))
    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []
    all_results_dict = {}
    total = 0.0

    for out_dataset in out_datasets:
        novel = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat', out_dataset,'out_scores.txt'))
        total += novel.shape[0]

        results = cal_metric(known, novel, method)

        all_results.append(results)
        all_results_dict[out_dataset] = results

    avg_results = compute_average_results(all_results)
    all_results_dict['All'] = avg_results
    print_results(avg_results, in_dataset, "All", name, method)
    
    return all_results_dict

def compute_in(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name):

    known_nat = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat','in_scores.txt'))
    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    known_nat_label = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat','in_labels.txt'))
    nat_in_cond = (known_nat>threshold).astype(np.float32)
    nat_correct = (known_nat_label[:,0] == known_nat_label[:,1]).astype(np.float32)
    known_nat_acc = np.mean(nat_correct)
    known_nat_fnr = np.mean((1.0 - nat_in_cond))
    known_nat_eteacc = np.mean(nat_correct * nat_in_cond)

    print('In-distribution performance:')
    print('FNR: {fnr:6.2f}, Acc: {acc:6.2f}, End-to-end Acc: {eteacc:6.2f}'.format(fnr=known_nat_fnr*100,acc=known_nat_acc*100,eteacc=known_nat_eteacc*100))
    return {'ACC':known_nat_acc * 100}

def compute_percls_in(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name):

    known_nat = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat','in_scores.txt'))
    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    known_nat_label = np.loadtxt(os.path.join(base_dir, exp_cat_name, in_dataset, auxiliary_dataset, model_arch, method, name, 'nat','in_labels.txt'))
    nat_in_cond = (known_nat>threshold).astype(np.float32)
    nat_correct = (known_nat_label[:,0] == known_nat_label[:,1]).astype(np.float32)
    known_nat_acc = np.mean(nat_correct)
    known_nat_fnr = np.mean((1.0 - nat_in_cond))
    known_nat_eteacc = np.mean(nat_correct * nat_in_cond)

    cls_acc_list = []
    if in_dataset == 'CIFAR-10':
        categories = 10
    elif in_dataset == 'CIFAR-100':
        categories = 100
    for cls in range(categories):
        cls_correct = ((known_nat_label[:,0] == known_nat_label[:,1]) & (known_nat_label[:,0] == cls)).astype(np.float32)
        cls_acc = np.sum(cls_correct)/np.sum((known_nat_label[:, 1] == cls).astype(np.float32))
        cls_acc_list.append(cls_acc)
    print(cls_acc_list)

    print('In-distribution performance:')
    print('FNR: {fnr:6.2f}, Acc: {acc:6.2f}, End-to-end Acc: {eteacc:6.2f}'.format(fnr=known_nat_fnr*100,acc=known_nat_acc*100,eteacc=known_nat_eteacc*100))
    return

def compute_adv_ood(base_dir, in_dataset, out_datasets, method, name, epsilon):

    known_nat = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))

    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    print('L_infty attack')
    print('epsilon: ', epsilon)

    print('nat_in vs. adv_out:')
    all_results = []

    total = 0.0

    for out_dataset in out_datasets:
        novel_adv = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/adv/{epsilon}/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset, epsilon=epsilon))

        total += novel_adv.shape[0]

        known = known_nat
        novel = novel_adv

        results = cal_metric(known, novel, method)

        all_results.append(results)

    avg_results = compute_average_results(all_results)
    print_results(avg_results, in_dataset, "All", name, method)

    return

def compute_corrupt_ood(base_dir, in_dataset, out_datasets, method, name):

    known_nat = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))

    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    print('Corruption attack')

    print('nat_in vs. adv_out:')
    all_results = []

    total = 0.0

    for out_dataset in out_datasets:
        novel_adv = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/corrupt/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset))

        total += novel_adv.shape[0]

        known = known_nat
        novel = novel_adv

        results = cal_metric(known, novel, method)

        all_results.append(results)

    avg_results = compute_average_results(all_results)
    print_results(avg_results, in_dataset, "All", name, method)

    return

def compute_adv_corrupt_ood(base_dir, in_dataset, out_datasets, method, name, epsilon):

    known_nat = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/nat/in_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name))

    known_nat_sorted = np.sort(known_nat)
    num_k = known_nat.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known_nat_sorted[round(0.05 * num_k)]

    print('compositional attack')

    print('nat_in vs. adv_out:')
    all_results = []

    total = 0.0

    for out_dataset in out_datasets:
        novel_adv = np.loadtxt('{base_dir}/{in_dataset}/{method}/{name}/adv_corrupt/{epsilon}/{out_dataset}/out_scores.txt'.format(base_dir=base_dir, in_dataset=in_dataset, method=method, name=name, out_dataset=out_dataset, epsilon=epsilon))

        total += novel_adv.shape[0]

        known = known_nat
        novel = novel_adv

        results = cal_metric(known, novel, method)

        all_results.append(results)

    avg_results = compute_average_results(all_results)
    print_results(avg_results, in_dataset, "All", name, method)

    return


if __name__ == '__main__':

    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100" or args.in_dataset =='CIFAR-10_subset':
        out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365', 'SVHN']
    elif args.in_dataset == "SVHN":
        out_datasets = ['LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365', 'CIFAR-10']
    if args.if_aux:
        auxiliary_dataset = args.auxiliary_dataset
    else:
        auxiliary_dataset = 'no_auxiliary'
    if not os.path.exists('OOD.xlsx'):
        create_excel()
    ood_results_dict = compute_traditional_ood(args.base_dir, args.exp_cat_name, args.in_dataset, auxiliary_dataset, args.model_arch, out_datasets, args.method, args.name)
    id_results_dict = compute_in(args.base_dir, args.exp_cat_name, args.in_dataset, auxiliary_dataset, args.model_arch, args.method, args.name)
    out_datasets.append('All')
    for out_dataset in out_datasets:
        write_excel(ood_results_dict[out_dataset],in_dataset = args.in_dataset, out_dataset = out_dataset, name = args.name, method = args.method)
    write_id_excel(results = id_results_dict, in_dataset=args.in_dataset, name= args.name, method = args.method)