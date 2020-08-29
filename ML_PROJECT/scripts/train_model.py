from __future__ import division, absolute_import, print_function

import argparse
from detect.util import get_data, get_model


def main(args):
    dataset='mnist'
    X_train, Y_train, X_test, Y_test = get_data()
    model = get_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    model.fit(
        X_train, Y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
    model.save('../data/model_%s.h5' % dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(epochs=10)
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    main(args)
