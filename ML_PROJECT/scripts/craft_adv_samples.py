from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from detect.util import get_data
from detect.attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method)

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {'eps': 0.300, 'eps_iter': 0.010}


def craft_one_type(sess, model, X, Y, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """
    dataset='mnist'
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS['eps'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
    elif attack in ['bim', 'bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS['eps'],
            eps_iter=ATTACK_PARAMS['eps_iter'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
        if attack == 'bim-a' or attack == 'bim':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        )
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/Adv_%s_%s.npy' % (dataset, attack), X_adv)


def main(args):
    dataset='mnist'
    assert args.attack in ['fgsm', 'bim', 'bim-a', 'bim-b', 'jsma', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim', bim-a', 'bim-b', or 'jsma'"
    assert os.path.isfile('../data/model_%s.h5' % dataset), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (dataset, args.attack))
    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
    model = load_model('../data/model_%s.h5' % dataset)
    _, _, X_test, Y_test = get_data()
    _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size, verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim','bim-a', 'bim-b', 'jsma']:
            craft_one_type(sess, model, X_test, Y_test, attack, args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(sess, model, X_test, Y_test, args.attack, args.batch_size)
    print('Adversarial samples crafted and saved to data/ subfolder.')
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim', bim-a', 'bim-b', 'jsma', or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
