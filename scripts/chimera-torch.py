#!/usr/bin/env python

from argparse import ArgumentParser

from _script_common import add_general_argument
from _script_common import validate_general_argument
from _script_common import add_feature_argument
from _script_common import validate_feature_argument
from _script_common import add_model_argument
from _script_common import validate_model_argument
from _train import add_training_io_argument
from _train import validate_training_io_argument
from _train import add_training_argument
from _train import validate_training_argument
from _train import train
from _predict import add_prediction_io_argument
from _predict import validate_prediction_io_argument
from _predict import predict
from _evaluate import add_evaluation_io_argument
from _evaluate import validate_evaluation_io_argument
from _evaluate import evaluate

'''
Command line arguments
'''

def parse_args():
    parser = ArgumentParser()
    subparser = parser.add_subparsers(help='<<subcommand help>>', dest='command')
    train_parser = subparser.add_parser('train', help='<<train help')
    train_general_parser = train_parser.add_argument_group('general')
    train_io_parser = train_parser.add_argument_group('io')
    train_feature_parser = train_parser.add_argument_group('feature')
    train_model_parser = train_parser.add_argument_group('model')
    train_training_parser = train_parser.add_argument_group('training')
    add_general_argument(train_general_parser)
    add_training_io_argument(train_io_parser)
    add_feature_argument(train_feature_parser)
    add_model_argument(train_model_parser)
    add_training_argument(train_training_parser)

    predict_parser = subparser.add_parser('predict', help='<<predict help>>')
    predict_general_parser = predict_parser.add_argument_group('general')
    predict_io_parser = predict_parser.add_argument_group('io')
    predict_feature_parser = predict_parser.add_argument_group('feature')
    add_general_argument(predict_general_parser)
    add_prediction_io_argument(predict_io_parser)
    add_feature_argument(predict_feature_parser)

    evaluate_parser = subparser.add_parser('evaluate', help='<<evaluate help>>')
    evaluate_general_parser = evaluate_parser.add_argument_group('general')
    evaluate_io_parser = evaluate_parser.add_argument_group('io')
    evaluate_feature_parser = evaluate_parser.add_argument_group('feature')
    add_general_argument(evaluate_general_parser)
    add_evaluation_io_argument(evaluate_io_parser)
    add_feature_argument(evaluate_feature_parser)

    args = parser.parse_args()
    if args.command == 'train':
        validate_general_argument(args, parser)
        validate_training_io_argument(args, parser)
        validate_feature_argument(args, parser)
        validate_model_argument(args, parser)
        validate_training_argument(args, parser)
    elif args.command == 'predict':
        validate_general_argument(args, parser)
        validate_prediction_io_argument(args, parser)
        validate_feature_argument(args, parser)
    elif args.command == 'evaluate':
        validate_general_argument(args, parser)
        validate_evaluation_io_argument(args, parser)
        validate_feature_argument(args, parser)
    return args

def main():
    args = parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate(args)

if __name__ == '__main__':
    main()
