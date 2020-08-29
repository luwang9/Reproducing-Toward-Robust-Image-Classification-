from __future__ import division, absolute_import, print_function

import os
import argparse
import warnings
import numpy as np
import cv2
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from detect.util import (get_data, get_mc_predictions, normalize)

def crop_resize(X,batch_size):
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    ls=[]
    for i in range(n_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        new_X = np.transpose(X_batch.squeeze(axis=-1),(1,2,0))
        #X_center = cv2.resize(new_X[4:24,4:24,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_center = cv2.resize(new_X[1:-1,1:-1,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_left_top = cv2.resize(new_X[:27,:27,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_left_bot = cv2.resize(new_X[1:,:27,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_right_top = cv2.resize(new_X[:27,1:,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_right_bot = cv2.resize(new_X[1:,1:,:],dsize=(28,28),interpolation=cv2.INTER_CUBIC)
        X_proc = np.asarray([X_center,X_left_top,X_left_bot,X_right_top,X_right_bot])
        if X_proc.ndim!=4:
            X_proc=np.expand_dims(X_proc,axis=-1)
        X_proc = np.transpose(X_proc,(0,3,1,2))
        ls.append(X_proc)
    X_proc=np.concatenate(ls,axis=1)
    return np.expand_dims(X_proc,axis=-1)

def preprocess_infer(X, preds, model, batch_size, C):
    X_proc = crop_resize(X,batch_size)
    label = np.zeros(preds.shape[0])
    ct = np.zeros(preds.shape[0])
    for i in range(5):
        x = X_proc[i]
        new_preds = model.predict_classes(x, verbose=0, batch_size=batch_size)
        same_or_not = new_preds==preds
        ct += same_or_not
    label[np.where(ct>=C)[0]] = 1

    return label

def detect_clean_adv(X, preds, uncert, model, C, H, L, batch_size):
    label = -1 * np.ones(preds.shape[0])
    #print(label.shape[0])
    label[np.where(uncert>H)[0]] = 0
    #middle_idx = np.where(label == -1)[0]
    #print(middle_idx.shape[0])
    label[np.where(uncert<L)[0]] = 1
    middle_idx = np.where(label == -1)[0]
    #print(middle_idx.shape[0])
    if middle_idx.shape[0] != 0:
        label[middle_idx] = preprocess_infer(X[middle_idx], preds[middle_idx], model, batch_size, C)
    return label

def detect_clean_adv_v2(X, preds, uncert, model, C, H, L, batch_size):
    label = -1 * np.ones(preds.shape[0])
    #print(label.shape[0])
    label[np.where(uncert>H)[0]] = 0
    #middle_idx = np.where(label == -1)[0]
    #print(middle_idx.shape[0])
    label[np.where(uncert<=H)[0]] = 1
    #middle_idx = np.where(label == -1)[0]
    #print(middle_idx.shape[0])
    #if middle_idx.shape[0] != 0:
    #    label[middle_idx] = preprocess_infer(X[middle_idx], preds[middle_idx], model, batch_size, C)
    return label
def main(args):
    attack=args.attack
    text_file = open("../stats/"+attack+"_stats.txt", "w")
    sd_start=args.sd_start
    num_sd=args.num_sd
    for sd in range(sd_start,sd_start+num_sd):
        print("seed: "+str(sd))
        with_norm=False
        batch_size=256
        np.random.seed(sd)
        idx0=np.random.choice(10000,5000)
        dataset='mnist'
        assert attack in ['fgsm', 'bim', 'bim-a', 'bim-b', 'jsma'], \
            "Attack parameter must be either 'fgsm', 'bim', bim-a', 'bim-b', 'jsma'"
        assert os.path.isfile('../data/model_%s.h5' % dataset), \
            'model file not found... must first train model using train_model.py.'
        assert os.path.isfile('../data/Adv_%s_%s.npy' % (dataset, attack)), \
            'adversarial sample file not found... must first craft adversarial ' \
            'samples using craft_adv_samples.py'
        print('Loading the data and model...')
        # Load the model
        model = load_model('../data/model_%s.h5' % dataset)
        # Load the dataset
        X_train, Y_train, X_test, Y_test = get_data()
        # Check attack type, select adversarial samples accordingly
        print('Loading adversarial samples...')
        X_test_adv = np.load('../data/Adv_%s_%s.npy' % (dataset, attack))



        ################################################ Table 1 ###########################################
        #Get half data from clean and adversarial images respectively and combine them
        X_test, Y_test, X_test_adv = X_test[idx0], Y_test[idx0], X_test_adv[idx0]
        X_test_all_un = np.concatenate((X_test, X_test_adv),axis=0)
        Y_test_all_un = np.concatenate((Y_test, Y_test),axis=0)
        # Check model accuracies on each sample type and then on combined undefended dataset
        for s_type, dt in zip(['normal', 'adversarial'], [X_test, X_test_adv]):
            _, acc = model.evaluate(dt, Y_test, batch_size=batch_size, verbose=0)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
        _, acc = model.evaluate(X_test_all_un, Y_test_all_un, batch_size=batch_size, verbose=0)
        print("Model accuracy on the combined undefended test set: %0.2f%%" % (100 * acc))
        
        
        ################################################ Table 2 ###########################################
        # Refine the normal and adversarial sets to only include samples,
        # for which the original version was correctly classified by the model.
        # Then, create detector label for clean as "1" and for adversarial as "0"
        preds_test = model.predict_classes(X_test, verbose=0, batch_size=batch_size)
        inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
        X_test = X_test[inds_correct]
        X_test_adv = X_test_adv[inds_correct]
        Y_test = Y_test[inds_correct]
        label_clean=np.ones(Y_test.shape[0])
        label_adv=np.zeros(Y_test.shape[0])

        # Combine the filtered dataset and detector label
        X_test_all_filtered = np.concatenate((X_test, X_test_adv),axis=0)
        Y_test_all_filtered = np.concatenate((Y_test, Y_test),axis=0)
        label_filtered = np.concatenate((label_clean, label_adv),axis=0)

        # Get Prediction + Bayesian uncertainty scores
        print('Getting Monte Carlo dropout variance predictions...')
        #pred_normal = get_mc_predictions(model, X_test, batch_size=batch_size)
        #pred_adv = get_mc_predictions(model, X_test_adv, batch_size=batch_size)
        pred_all_filtered = get_mc_predictions(model, X_test_all_filtered, batch_size=batch_size)
        #uncerts_normal = pred_normal.var(axis=0).mean(axis=1)
        #uncerts_adv = pred_adv.var(axis=0).mean(axis=1)
        uncerts_all = pred_all_filtered.var(axis=0).mean(axis=1)
        if with_norm:
            ## Z-score the uncertainty
            uncerts_all = normalize(uncerts_all)
        #uncerts_all = np.concatenate((uncerts_normal, uncerts_adv),axis=0)
        #class_normal = pred_normal.mean(axis=0).argmax(axis=1)
        #class_adv = pred_adv.mean(axis=0).argmax(axis=1)
        #preds_test_all_filtered = np.concatenate((class_normal, class_adv),axis=0)
        preds_test_all_filtered = pred_all_filtered.mean(axis=0).argmax(axis=1)
        
        # Detector Parameters to be fine-tuned in experiments
        params = {
            'fgsm': {'H': 0.002,  'L': 0.000003, 'C': 5},
            'bim':  {'H': 0.0022, 'L': 0.0012,   'C': 5},
            'jsma': {'H': 0.01,   'L': 0.003,    'C': 5}
        }
        start_H, start_L, start_C = params[attack]["H"], params[attack]["L"], params[attack]["C"]
        H_range, L_range, C_range = 0, 0, 1
        H_step, L_step, C_step = 1e-4, 5e-6, 1

        threshold_ls=[]
        estimator = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=3, splitter='best',random_state=None)
        X = np.expand_dims(uncerts_all,axis=-1)
        Y = label_filtered
        estimator.fit(X, Y)
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        threshold = estimator.tree_.threshold
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        for i in range(n_nodes):
            if not is_leaves[i]:
                threshold_ls.append(threshold[i])
        start_H_d=max(threshold_ls)
        start_L_d=min(threshold_ls)

        #Detection
        for C in range(start_C, min(start_C+C_range,5)+C_step, C_step):
            for sad in range(2):
                H=[start_H_d,start_H][sad]
                L=[start_L_d,start_L][sad]
                label_pred=detect_clean_adv(X_test_all_filtered, preds_test_all_filtered, uncerts_all, model, C, H, L, batch_size)
                #Detection Evaluation
                CM=confusion_matrix(label_filtered, label_pred)
                TN = CM[0][0]
                FN = CM[1][0]
                TP = CM[1][1]
                FP = CM[0][1]
                ACC_DETECT = (TP+TN)/(TP+TN+FP+FN)
                res_detect = "Seed: " + str(sd) + ", C: " + str(C) + ", H: " + str(H)+ ", L: " + str(L) \
                                + "\nDetection Acc: " + str(ACC_DETECT*100)[:4] \
                                + "%, False Negative: " + str(FN) + ", False Positive: " + str(FP) \
                                + ", True Negative: " + str(TN) + ", True Positive: " + str(TP)+"."
                print(res_detect)
                text_file.write(res_detect+"\n")

                ################################################ Table 3 ###########################################
                #Only get images reported as clean
                idx_clean_reported=np.where(label_pred)[0]
                X_test_all_def = X_test_all_filtered[idx_clean_reported]
                Y_test_all_def = Y_test_all_filtered[idx_clean_reported]
                label_def = label_filtered[idx_clean_reported]
                num_all=label_def.shape[0]

                # Reclassification
                print('Computing final model predictions...')
                preds_test_all_def = model.predict_classes(X_test_all_def, verbose=0, batch_size=batch_size)
                inds_correct_def = np.where(preds_test_all_def == Y_test_all_def.argmax(axis=1))[0]
                inds_incorrect_def = np.where(preds_test_all_def != Y_test_all_def.argmax(axis=1))[0]

                # Reclassification Evaluation 
                num_clean_correct = np.argwhere(label_def[inds_correct_def] == 1).shape[0]
                num_clean_incorrect = np.argwhere(label_def[inds_incorrect_def] == 1).shape[0]
                num_adv_correct = np.argwhere(label_def[inds_correct_def] == 0).shape[0]
                num_adv_incorrect = np.argwhere(label_def[inds_incorrect_def] == 0).shape[0]
                clean_correct_acc = num_clean_correct/num_all
                clean_incorrect_acc = num_clean_incorrect/num_all
                adv_correct_acc = num_adv_correct/num_all
                adv_incorrect_acc = num_adv_incorrect/num_all
                total_acc = clean_correct_acc + adv_correct_acc
                res_reclf = "Reclassification Acc: " + str(total_acc*100)[:4] \
                + "%, Clean Correct Acc: " + str(clean_correct_acc*100)[:4] \
                + "%, Clean Incorrect Acc: " + str(clean_incorrect_acc*100)[:4] \
                + "%, Adv Correct Acc: " + str(adv_correct_acc*100)[:4] \
                + "%, Adv Incorrect Acc: " + str(adv_incorrect_acc*100)[:4] + "%."
                print(res_reclf)
                text_file.write(res_reclf+"\n")


    text_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim', bim-a', 'bim-b', 'jsma', or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-s', '--sd_start',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-n', '--num_sd',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(sd_start=0)
    parser.set_defaults(num_sd=1)
    args = parser.parse_args()
    main(args)
