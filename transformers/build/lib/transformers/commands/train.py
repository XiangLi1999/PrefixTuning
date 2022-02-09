import os
from argparse import ArgumentParser, Namespace

from transformers import SingleSentenceClassificationProcessor as Processor
from transformers import TextClassificationPipeline, is_tf_available, is_torch_available
from transformers.commands import BaseTransformersCLICommand

from ..utils import logging


if not is_tf_available() and not is_torch_available():
    raise RuntimeError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TF training parameters
USE_XLA = False
USE_AMP = False


def train_command_factory(args: Namespace):
    """
    Factory function used to instantiate training command from provided command line arguments.
    :return: TrainCommand
    """
    return TrainCommand(args)


class TrainCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli
        :param parser: Root parser to register command-specific arguments
        :return:
        """
        train_parser = parser.add_parser("train", help="CLI tool to train a model on a task.")

        train_parser.add_argument(
            "--train_data",
            type=str,
            required=True,
            help="path to train (and optionally evaluation) dataset as a csv with "
            "tab separated labels and sentences.",
        )
        train_parser.add_argument(
            "--column_label", type=int, default=0, help="Column of the dataset csv file with example labels."
        )
        train_parser.add_argument(
            "--column_text", type=int, default=1, help="Column of the dataset csv file with example texts."
        )
        train_parser.add_argument(
            "--column_id", type=int, default=2, help="Column of the dataset csv file with example ids."
        )
        train_parser.add_argument(
            "--skip_first_row", action="store_true", help="Skip the first row of the csv file (headers)."
        )

        train_parser.add_argument("--validation_data", type=str, default="", help="path to validation dataset.")
        train_parser.add_argument(
            "--validation_split",
            type=float,
            default=0.1,
            help="if validation dataset is not provided, fraction of train dataset " "to use as validation dataset.",
        )

        train_parser.add_argument("--output", type=str, default="./", help="path to saved the trained model.")

        train_parser.add_argument(
            "--task", type=str, default="text_classification", help="Task to train the model on."
        )
        train_parser.add_argument(
            "--model", type=str, default="bert-base-uncased", help="Model's name or path to stored model."
        )
        train_parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")
        train_parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation.")
        train_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
        train_parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam optimizer.")
        train_parser.set_defaults(func=train_command_factory)

    def __init__(self, args: Namespace):
        self.logger = logging.get_logger("transformers-cli/training")

        self.framework = "tf" if is_tf_available() else "torch"

        os.makedirs(args.output, exist_ok=True)
        self.output = args.output

        self.column_label = args.column_label
        self.column_text = args.column_text
        self.column_id = args.column_id

        self.logger.info("Loading {} pipeline for {}".format(args.task, args.model))
        if args.task == "text_classification":
            self.pipeline = TextClassificationPipeline.from_pretrained(args.model)
        elif args.task == "token_classification":
            raise NotImplementedError
        elif args.task == "question_answering":
            raise NotImplementedError

        self.logger.info("Loading dataset from {}".format(args.train_data))
        self.train_dataset = Processor.create_from_csv(
            args.train_data,
            column_label=args.column_label,
            column_text=args.column_text,
            column_id=args.column_id,
            skip_first_row=args.skip_first_row,
        )
        self.valid_dataset = None
        if args.validation_data:
            self.logger.info("Loading validation dataset from {}".format(args.validation_data))
            self.valid_dataset = Processor.create_from_csv(
                args.validation_data,
                column_label=args.column_label,
                column_text=args.column_text,
                column_id=args.column_id,
                skip_first_row=args.skip_first_row,
            )

        self.validation_split = args.validation_split
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon

    def run(self):
        if self.framework == "tf":
            return self.run_tf()
        return self.run_torch()

    def run_torch(self):
        raise NotImplementedError

    def run_tf(self):
        self.pipeline.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            validation_split=self.validation_split,
            learning_rate=self.learning_rate,
            adam_epsilon=self.adam_epsilon,
            train_batch_size=self.train_batch_size,
            valid_batch_size=self.valid_batch_size,
        )

        # Save trained pipeline
        self.pipeline.save_pretrained(self.output)
