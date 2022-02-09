import inspect
import json
import math
import os
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from nltk import word_tokenize
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
import random

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .file_utils import is_datasets_available, is_torch_tpu_available
from .integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
from .modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    EvaluationStrategy,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    distributed_broadcast_scalars,
    distributed_concat,
    nested_concat,
    nested_numpify,
    nested_xla_mesh_reduce,
    set_seed,
)
from .training_args import TrainingArguments
from .utils import logging


_use_native_amp = False
_use_apex = False
EPS = 1e-12
INIT_GUMBEL_TEMP = 5.0

control_lst = ['positive', 'negative', 'neutral']
Control_Temp = {'positive': 3967, 'negative':4633, 'neutral':8500}
control_Map = [torch.LongTensor([3967]), torch.LongTensor([4633]), torch.LongTensor([8500])]
sst_lst = [(0, 2), (1, 3), (4,)]
sst_standard = ["positive", "negative", "very positive", "very negative", "neutral"]
# Control_?Map = {j:i for i, j in enumerate(control_lst)}

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from .file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune


logger = logging.get_logger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def helper_token2bpe(offsets):
    full_lst = []
    for example_offset in offsets:
        bpe2token = []
        token2bpe = []
        token_idx = -1
        # print(example_offset)
        for bpe_idx, (a,b) in enumerate(example_offset):
            # print(token2bpe, a, b, bpe_idx)
            if b - a > 0:
                if a == 0:
                    # new token
                    token_idx += 1
                    bpe2token.append(token_idx)
                    token2bpe.append([])
                    token2bpe[-1].append(bpe_idx)
                else:
                    # prev token.
                    bpe2token.append(token_idx)
                    token2bpe[-1].append(bpe_idx)
            else:
                bpe2token.append(None)
        full_lst.append((bpe2token, token2bpe))
    return full_lst

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer_Prefix:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of :class:`~transformers.TrainingArguments`
            with the ``output_dir`` set to a directory named `tmp_trainer` in the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or
            :obj:`eval_dataset`. Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is
            provided, an instance of :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        tb_writer (:obj:`SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
        kwargs:
            Deprecated keyword arguments.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        model_gpt2 : Optional[PreTrainedModel] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        discri_model: Optional[PreTrainedModel] = None,
        discri_tokenizer:Optional["PreTrainedTokenizerBase"] = None,
        dataless_sample_size:Optional[int] = None,
        dataless_sample_length:Optional[int] = None,
        dataless_control_type:Optional[int] = None,
        dataless_usebaseline:Optional[bool] = None,
        discri_labels:Optional[list]= None,
        reverse_kl:Optional[str]= None,
        forward_kl: Optional[str] = None,
        sample_from_gpt:Optional[bool] = False,
        gumbel: Optional[bool] = None,
        replay_buffer: Optional[bool] = None,
        adaptive_data:Optional[bool] = True,
        task_mode: Optional[str] = None,
        use_dropout: Optional[bool] = False,
        both_tune: Optional[bool] = False,
        distill: Optional[bool] = False,
        matching_objective:Optional[str]= None,
        finetuned_gpt2: Optional[PreTrainedModel] = None,
        **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        assert (
            model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument."
        assert model_init is None
        self.model = model.to(args.device) if model is not None else None
        self.gpt2 = model_gpt2.to(args.device) if model_gpt2 is not None else None
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.task_mode = task_mode
        self.use_dropout = use_dropout

        self.curr_best_eval = 10000000.
        self.both_tune = both_tune

        self.distill = distill
        if self.distill:
            self.matching_objective = matching_objective
            self.finetuned_gpt2 = finetuned_gpt2

        # for dataless.
        if dataless_control_type is not None:
            self.gumbel = gumbel
            if self.gumbel:
                self.adaptive_data = False
                self.replay_buffer = False
            else:
                self.adaptive_data = adaptive_data
                self.replay_buffer = replay_buffer
            if self.gumbel:
                self.gumbel_temperature = INIT_GUMBEL_TEMP

            if self.replay_buffer:
                self.buffer_lst = []
            self.discri_model = discri_model
            self.discri_tokenizer = discri_tokenizer

            self.dataless_sample_size =  dataless_sample_size
            self.sample_from_gpt = sample_from_gpt
            print('sample from gpt2', self.sample_from_gpt)
            self.reverse_kl = (reverse_kl=='yes')
            self.forward_kl = (forward_kl == 'yes')
            print('forward/reverse', self.forward_kl, self.reverse_kl)
            assert (self.reverse_kl ^ self.forward_kl)
            self.dataless_sample_length = dataless_sample_length
            self.dataless_control_type = dataless_control_type
            self.dataless_usebaseline = dataless_usebaseline
            self.discri_labels = discri_labels
            if self.tokenizer is not None and self.discri_labels is not None:
                print(self.discri_labels)
                self.discri_labels_code = self.tokenizer(self.discri_labels, return_tensors="pt",
                                                         is_split_into_words=True, add_special_tokens=False)['input_ids']
                self.discri_labels_code = self.discri_labels_code.view(-1).\
                    to(self.args.device).unsqueeze(0).split(1, dim=1)
                print(self.discri_labels_code)
            else:
                self.discri_labels_code = None
        else:
            self.dataless_sample_length = 60
            self.discri_labels = discri_labels
            print(self.discri_labels)
            print('training with data, DEFINITELY CHECK HERE')
            if self.tokenizer is not None and self.discri_labels is not None:
                print(self.discri_labels)
                self.discri_labels_code = self.tokenizer(self.discri_labels, return_tensors="pt",
                                                         is_split_into_words=True, add_special_tokens=False)[
                    'input_ids']
                self.discri_labels_code = self.discri_labels_code.view(-1). \
                    to(self.args.device).unsqueeze(0).split(1, dim=1)
                print(self.discri_labels_code)
            else:
                self.discri_labels_code = None

        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        self.tb_writer = tb_writer
        self.log_history = []
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a future version. Use `args.prediction_loss_only` instead.",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        if tb_writer is None and is_tensorboard_available() and self.is_world_process_zero():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        self.global_step = None
        self.epoch = None
        self.total_flos = None
        if self.args.fp16 and _use_native_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        if self.args.label_names is None:
            self.args.label_names = (
                ["start_positions, end_positions"]
                if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
                else ["labels"]
            )

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description}don't have a corresponding argument in `{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
        dataset.set_format(type=dataset.format["type"], columns=columns)

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=np.random.seed(self.args.seed)
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=np.random.seed(self.args.seed)
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`test_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed.
        """
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=np.random.seed(self.args.seed)
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            # DEBUG
            # optimizer_grouped_parameters = [
            #     {
            #         "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            #         "weight_decay": self.args.weight_decay,
            #     },
            #     {
            #         "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            #         "weight_decay": 0.0,
            #     },
            # ]
            # self.optimizer = AdamW(
            #     optimizer_grouped_parameters,
            #     lr=self.args.learning_rate,
            #     betas=(self.args.adam_beta1, self.args.adam_beta2),
            #     eps=self.args.adam_epsilon,
            # )


            if self.both_tune:
                print('Optimizing parameters for both prefixtune and the finetune. ')
                full_named_lst = list(self.model.named_parameters()) + list(self.gpt2.named_parameters())
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in full_named_lst if
                                   (not any(nd in n for nd in no_decay)) and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in full_named_lst if
                                   any(nd in n for nd in no_decay) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

            else:
                optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
                ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )


            # for n, p in self.model.named_parameters():
            #     print(n,p.requires_grad)
            print(self.optimizer.state_dict())
        if self.lr_scheduler is None:
            # URGENT!!!
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
            # URGENT!!!
            # self.lr_scheduler = get_constant_schedule_with_warmup(
            #     self.optimizer, num_warmup_steps=self.args.warmup_steps
            # )



    def setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        `here <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if hasattr(self, "_setup_wandb"):
            warnings.warn(
                "The `_setup_wandb` method is deprecated and won't be called in a future version, define `setup_wandb` in your subclass.",
                FutureWarning,
            )
            return self._setup_wandb()

        if self.is_world_process_zero():
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            try:
                combined_dict = {**self.model.config.to_dict(), **self.args.to_sanitized_dict()}
            except AttributeError:
                # in case the model has no config
                combined_dict = {**self.args.to_sanitized_dict()}
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"), config=combined_dict, name=self.args.run_name
            )
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def setup_comet(self):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE:
                (Optional): str - "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME:
                (Optional): str - Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY:
                (Optional): str - folder to use for saving offline experiments when `COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment,
        see `here <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__
        """
        if self.is_world_master():
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            args = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            experiment = None
            if comet_mode == "ONLINE":
                experiment = comet_ml.Experiment(**args)
                logger.info("Automatic Comet.ml online logging enabled")
            elif comet_mode == "OFFLINE":
                args["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**args)
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            if experiment is not None:
                experiment._set_model_graph(self.model, framework="transformers")
                experiment._log_parameters(self.args, prefix="args/", framework="transformers")
                experiment._log_parameters(self.model.config, prefix="config/", framework="transformers")

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.
        """
        return len(dataloader.dataset)

    def _setup_loggers(self):
        if self._loggers_initialized:
            return
        if is_wandb_available():
            self.setup_wandb()
        elif os.environ.get("WANDB_DISABLED") != "true":
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        if is_comet_available():
            self.setup_comet()
        elif os.environ.get("COMET_MODE") != "DISABLED":
            logger.info(
                "To use comet_ml logging, run `pip/conda install comet_ml` "
                "see https://www.comet.ml/docs/python-sdk/huggingface/"
            )
        self._loggers_initialized = True

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """ HP search setup code """
        if self.hp_search_backend is None or trial is None:
            return
        params = self.hp_space(trial) if self.hp_search_backend == HPSearchBackend.OPTUNA else trial
        for key, value in params.items():
            if not hasattr(self.args, key):
                raise AttributeError(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
                )
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info("Trial:", trial.params)

    def _report_to_hp_search(
        self, trial: Union["optuna.Trial", Dict[str, Any]], epoch: int, metrics: Dict[str, float]
    ):
        if self.hp_search_backend is None or trial is None:
            return
        self.objective = self.compute_objective(metrics)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            trial.report(self.objective, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            if self.global_step % self.args.save_steps == 0:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)

    def _tune_save_checkpoint(self):
        if not self.use_tune_checkpoints:
            return
        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
            self.save_model(output_dir)
            if self.is_world_master():
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], padding_value: Optional[int]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors

        length_of_first = examples[0].size(1)
        are_tensors_same_length = all(x.size(1) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.cat(examples, dim=0)
        else:
            examples = [x.transpose(0,1) for x in examples]
            temp = pad_sequence(examples, batch_first=True, padding_value=padding_value) #b*maxseqlen*bsz
            temp = temp.transpose(1,2) # bsz, maxseqlen
            bsz2, bsz, seqlen = temp.shape
            temp = temp.reshape(bsz2*bsz, seqlen)
            return temp

    def get_codes(self, selected_sents):
        # select keywords and skip the stopwords.
        device = self.discri_model.device
        print(selected_sents)
        # TODO.
        result_sent = []
        result_cand = []
        result_idx = []
        for j in selected_sents.tolist():
            print(j)
            temp_sent = self.tokenizer.decode(j)
            print(temp_sent)
            sent_token = temp_sent.split()
            result_sent.append(sent_token)
            cand_idx = random.choices(list(range(len(sent_token))), k=1)[0]
            cand_word = sent_token[cand_idx]
            print(cand_idx, cand_word)
            result_cand.append(cand_word)
            result_idx.append(cand_idx)



        temp_input = self.discri_tokenizer(result_sent, return_tensors="pt", padding=True,
              is_split_into_words=True, return_offsets_mapping=True, add_special_tokens=True)
        bsz, seqlen = temp_input['input_ids'].shape
        mask_input = temp_input['attention_mask']
        # incorporate a list of stop words.
        full_lst = helper_token2bpe(temp_input["offset_mapping"])
        bpe2token, token2bpe = zip(*full_lst)

        # get mask.
        bpe_idx = torch.LongTensor([token2bpe[bi][idx_][-1] for bi, idx_ in enumerate(result_idx)]).view(bsz, 1)
        mask = torch.zeros(bsz, seqlen).scatter_(1, bpe_idx, 1).bool()

        # print(temp_input)

        outputs_model = self.discri_model(temp_input['input_ids'].to(device), attention_mask=mask_input.to(device), output_hidden_states=True)
        last_hidden_states = outputs_model.last_hidden_state
        result_emb = last_hidden_states[mask].unsqueeze(1)
        return result_emb, result_cand


    def get_dataless_input(self, attribute_type=2, sample_size=10, sample_from_gpt=False, input_ids_prompt=None):
        # TODO
        # sample a attribute:
        device = self.model.device
        full_results = []
        control_codes = []
        sst_codes = []
        prompt_codes = []
        if attribute_type == 0:

            for a in range(len(self.discri_labels)):
                sst_label = self.discri_labels[a]
                control_code = self.discri_labels_code[a]
                control_codes += [control_code] * sample_size
                sst_codes += [a] * sample_size
                if not sample_from_gpt:
                    prompt = self.model.get_prompt(control_code, self.gpt2)
                    prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in
                              prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
                else:
                    prompt = None
                # print(len(prompt), prompt[0].shape)
                prompt_codes.append(prompt)

            prompt_codes = list(zip(*prompt_codes))
            # print(len(prompt_codes), len(prompt_codes[0]), prompt_codes[0][0].shape)
            prompt_full = []
            for prompt_c in prompt_codes:
                # print(len(prompt_c), prompt_c[0].shape, prompt_c[1].shape)
                prompt_c = torch.cat(prompt_c, dim=1)
                prompt_full.append(prompt_c)

            full_results = self.gpt2.generate(input_ids=input_ids_prompt,
                                              emb_match=None,
                                              control_code=None,
                                              past_key_values=prompt_full,
                                              max_length=self.dataless_sample_length,
                                              temperature=1.0,
                                              top_k=0,
                                              top_p=0.9,
                                              repetition_penalty=1.0,
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              do_sample=True,
                                              num_return_sequences=sample_size * len(self.discri_labels),
                                              # bad_words_ids=[[628], [198]] if True else None,
                                              use_cache=True)

            sst_codes = torch.LongTensor(sst_codes)
            control_codes = torch.cat(control_codes, dim=0)
            # print('look here ' * 10)
            # print(full_results)
            # labels = full_results.clone()
            # if self.tokenizer.eos_token_id is not None:
            #     mask = (labels == self.tokenizer.eos_token_id)
            #     mask_cumsum = mask.cumsum(1)
            #     mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
            #     labels[mask] = -100
            # return {'input_ids': full_results, 'control_code': control_codes, 'labels': labels, 'sst_codes': sst_codes}
            return {'input_ids': full_results, 'control_code': control_codes, 'sst_codes': sst_codes}



            # # discrete attribute
            # for a in range(2): # SST-5
            #     control_code = control_Map[a].unsqueeze(0).to(device)
            #     control_codes += [control_code] * sample_size
            #     sst_codes += [a] * sample_size
            #     prompt = self.model.get_prompt(control_code, self.gpt2)
            #     prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in prompt] # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
            #     results = self.gpt2.generate(input_ids=None,
            #                                  emb_match=None,
            #                                  control_code=None,
            #                                  past_key_values=prompt,
            #                                  max_length=self.dataless_sample_length,
            #                                  temperature=1.0,
            #                                  top_k=0,
            #                                  top_p=0.9,
            #                                  pad_token_id = self.tokenizer.pad_token_id,
            #                                  repetition_penalty=1.0,
            #                                  do_sample=True,
            #                                  num_return_sequences=sample_size,
            #                                  use_cache=True)
            #     full_results.append(results)
            #
            # sst_codes = torch.LongTensor(sst_codes)
            # full_results = self._tensorize_batch(full_results, self.tokenizer.eos_token_id)
            # # print(torch.abs(full_results2 - full_results).sum())
            # control_codes = torch.cat(control_codes, dim=0)
            # labels = full_results.clone()
            # if self.tokenizer.eos_token_id is not None:
            #     mask = (labels == self.tokenizer.eos_token_id)
            #     mask_cumsum = mask.cumsum(1)
            #     mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
            #     labels[mask] = -100
            # return {'input_ids':full_results, 'control_code':control_codes, 'labels':labels, 'sst_codes':sst_codes}

        elif attribute_type == 2 or attribute_type == 3:
            # discrete attribute
            for a in range(len(self.discri_labels)):
                sst_label = self.discri_labels[a]
                control_code = self.discri_labels_code[a]
                control_codes += [control_code] * sample_size
                sst_codes += [a] * sample_size
                if not sample_from_gpt:
                    prompt = self.model.get_prompt(control_code, self.gpt2)
                    prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in
                              prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
                else:
                    prompt = None
                # print(len(prompt), prompt[0].shape)
                prompt_codes.append(prompt)

            prompt_codes = list(zip(*prompt_codes))
            # print(len(prompt_codes), len(prompt_codes[0]), prompt_codes[0][0].shape)
            prompt_full = []
            for prompt_c in prompt_codes:
                # print(len(prompt_c), prompt_c[0].shape, prompt_c[1].shape)
                prompt_c = torch.cat(prompt_c, dim=1)
                prompt_full.append(prompt_c)



            full_results = self.gpt2.generate(input_ids=input_ids_prompt,
                                         emb_match=None,
                                         control_code=None,
                                         past_key_values=prompt_full,
                                         max_length=self.dataless_sample_length,
                                         temperature=1.0,
                                         top_k=0,
                                         top_p=0.9,
                                         repetition_penalty=1.0,
                                         do_sample=True,
                                         num_return_sequences=sample_size * len(self.discri_labels),
                                         bad_words_ids = [[628],[198]] if True else None,
                                         use_cache=True)
            # full_results.append(results)

            sst_codes = torch.LongTensor(sst_codes)
            # full_results = self._tensorize_batch(full_results, self.tokenizer.eos_token_id)

            control_codes = torch.cat(control_codes, dim=0)
            # labels = full_results.clone()
            # if self.tokenizer.eos_token_id is not None:
            #     mask = (labels == self.tokenizer.eos_token_id)
            #     mask_cumsum = mask.cumsum(1)
            #     mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
            #     labels[mask] = -100
            # return {'input_ids': full_results, 'control_code': control_codes, 'labels': labels, 'sst_codes': sst_codes}
            return {'input_ids': full_results, 'control_code': control_codes, 'sst_codes': sst_codes}

        elif attribute_type == 5:
            full_results = []
            control_word_lst = ['happiness', 'split', 'python', 'humor', 'advance', 'enjoyable']
            for control_word in control_word_lst:
                control_words = [control_word]

                control_code = self.tokenizer(control_words, return_tensors="pt", padding=True,
                                           is_split_into_words=True, add_special_tokens=False)['input_ids'].to(self.args.device)
                print(control_code)
                bsz, seqlen = control_code.shape
                prompt = self.model.get_prompt(control_code, self.gpt2)
                prompt_full = [x.expand(-1, sample_size, -1, -1, -1) for x in
                          prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)

                results = self.gpt2.generate(input_ids=input_ids_prompt,
                                                  emb_match=None,
                                                  control_code=None,
                                                  past_key_values=prompt_full,
                                                  max_length=self.dataless_sample_length,
                                                  temperature=1.0,
                                                  top_k=0,
                                                  top_p=0.9,
                                                  repetition_penalty=1.0,
                                                  do_sample=True,
                                                  num_return_sequences=sample_size,
                                                  bad_words_ids=[[628], [198]] if True else None,
                                                  use_cache=True)

                full_results.append(results)

            full_results = self._tensorize_batch(full_results, self.tokenizer.eos_token_id)

            return {'input_ids': full_results, 'control_code': None, 'sst_codes': None}

        # elif attribute_type == 2 or attribute_type == 3:
        #     # discrete attribute
        #     for a in range(len(self.discri_labels)):
        #         sst_label = self.discri_labels[a]
        #         control_code = self.discri_labels_code[a]
        #         control_codes += [control_code] * sample_size
        #         sst_codes += [a] * sample_size
        #         if not sample_from_gpt:
        #             prompt = self.model.get_prompt(control_code, self.gpt2)
        #             prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in
        #                       prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
        #         else:
        #             prompt = None
        #
        #         results = self.gpt2.generate(input_ids=None,
        #                                      emb_match=None,
        #                                      control_code=None,
        #                                      past_key_values=prompt,
        #                                      max_length=self.dataless_sample_length,
        #                                      temperature=1.0,
        #                                      top_k=0,
        #                                      top_p=0.9,
        #                                      repetition_penalty=1.0,
        #                                      do_sample=True,
        #                                      num_return_sequences=sample_size,
        #                                      bad_words_ids = [[628],[198]] if True else None,
        #                                      use_cache=True)
        #         full_results.append(results)
        #
        #     sst_codes = torch.LongTensor(sst_codes)
        #     full_results = self._tensorize_batch(full_results, self.tokenizer.eos_token_id)
        #
        #     control_codes = torch.cat(control_codes, dim=0)
        #     labels = full_results.clone()
        #     if self.tokenizer.eos_token_id is not None:
        #         mask = (labels == self.tokenizer.eos_token_id)
        #         mask_cumsum = mask.cumsum(1)
        #         mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
        #         labels[mask] = -100
        #     return {'input_ids': full_results, 'control_code': control_codes, 'labels': labels, 'sst_codes': sst_codes}
        elif attribute_type == 1:
            # continuous attribute
            # sample some attributes from Gaussian? this is wrong for sure.
            selected_sent = self.gpt2.generate(input_ids=None,
                                               emb_match=None,
                                               control_code=None,
                                               past_key_values=None,
                                               max_length=self.dataless_sample_length,
                                               temperature=1.0,
                                               top_k=0,
                                               top_p=0.9,
                                               repetition_penalty=1.0,
                                               do_sample=True,
                                               num_return_sequences=sample_size,
                                               use_cache=True)

            control_embs, control_word = self.get_codes(selected_sent)
            prompt = self.model.get_prompt((control_embs, control_word), self.gpt2)
            print('hello world')
            print(control_embs.shape, control_word)
            full_results = self.gpt2.generate(
                input_ids=None,
                emb_match=None,
                control_code=None,
                past_key_values=prompt,
                max_length=self.dataless_sample_length,
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=sample_size)

            labels = full_results.clone()
            if self.tokenizer.eos_token_id is not None:
                mask = (labels == self.tokenizer.eos_token_id)
                mask_cumsum = mask.cumsum(1)
                mask = (mask & (mask_cumsum != 1) & (mask_cumsum != 2))
                labels[mask] = -100

            return {'input_ids':full_results, 'control_embs':control_embs, 'control_word': control_word,
                    'labels':labels}



    def cleanup_samples(self, inputs):
        inputs['sst_codes'] = inputs['sst_codes'].tolist()
        inputs['control_code'] = inputs['control_code'].tolist()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_text = [self.tokenizer.decode(ll[1:], clean_up_tokenization_spaces=True) for ll in
                      inputs['input_ids'].tolist()]
        inputs['sents'] = []
        to_del = []
        for idx_, text in enumerate(input_text):
            text_idx = text.find(self.tokenizer.eos_token)
            if text_idx < 0:
                text_idx2 = text.rfind('.')
                text_idx2 = max(text_idx2, text.rfind('!'))
                text_idx2 = max(text_idx2, text.rfind('?'))
                if text_idx2 < 0:
                    # inputs['sents'].append(text)
                    to_del.append(idx_)
                else:
                    inputs['sents'].append(text[:text_idx2 + 1])
            else:
                inputs['sents'].append(text[:text_idx])

        for idx_ in reversed(to_del):
            # print(idx_)
            del inputs['sst_codes'][idx_]
            del inputs['control_code'][idx_]

        # update input_ids based on the clean up update.


        input_2 = self.tokenizer(inputs['sents'], return_tensors="pt", padding=True,
                        is_split_into_words=False, return_offsets_mapping=False, add_special_tokens=True)
        # append BOS to input_ids.
        # print(input_2)
        input_ids = input_2['input_ids']
        # print(input_ids.shape)
        bos = torch.LongTensor([self.tokenizer.bos_token_id]).unsqueeze(0).expand(len(inputs['sents']), -1)
        # bos_att = torch.LongTensor([1]).unsqueeze(0).expand(len(inputs['sents']), -1)
        # attn = torch.cat([bos_att, input_2['attention_mask']], dim=1)
        input_ids = torch.cat([bos, input_ids], dim=1)
        # print(input_ids.shape, input_2['attention_mask'].shape, attn.shape, attn.dtype, input_2['attention_mask'].dtype)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # print(input_ids)

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['control_code'] = torch.tensor(inputs['control_code'])
        inputs['sst_codes'] = torch.tensor(inputs['sst_codes'])
        # inputs['attention_mask'] = attn
        return inputs

    def train_dataless(self, model_path: Optional[str] = None,
                       trial: Union["optuna.Trial", Dict[str, Any]] = None,
                       verbose: Optional[bool] = False):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        # train_dataloader = self.get_train_dataloader()
        # num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        # num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        # num_update_steps_per_epoch = 5
        num_update_steps_per_epoch = 15000
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                # print(model, model.module)
                if self.args.n_gpu > 1:
                    self.total_flos = getattr(model.module.config, "total_flos", 0)
                else:
                    self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):

            #

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(range(num_update_steps_per_epoch), desc="Iteration", disable=disable_tqdm)
            # for step, inputs in enumerate(epoch_iterator):
            for step in range(num_update_steps_per_epoch):

                # Skip past any already trained steps if resuming training
                # if steps_trained_in_current_epoch > 0:
                #     steps_trained_in_current_epoch -= 1
                #     epoch_pbar.update(1)
                #     continue

                with torch.no_grad():
                    inputs = self.get_dataless_input(attribute_type=self.dataless_control_type,
                                                     sample_size=self.dataless_sample_size,
                                                     sample_from_gpt=self.sample_from_gpt)


                    ###############
                    if verbose and step % 10 == 9:
                        with torch.no_grad():
                            # print(inputs['input_ids'])
                            # print(inputs['labels'])
                            if self.sample_from_gpt:
                                print('sampling from q distribution for test; in training loop, we sample from GPT2.')
                                inputs_ = self.get_dataless_input(attribute_type=self.dataless_control_type,
                                                                  sample_size=self.dataless_sample_size,
                                                                  sample_from_gpt=False)
                                # TODO.

                                inputs_['sents'] = [self.tokenizer.decode(ll[1:]) for ll in
                                                    inputs_['input_ids'].tolist()]

                                for ii, ll in enumerate(inputs_['sents']):
                                    if ii % int(len(inputs['sents']) / len(self.discri_labels)) == 0:
                                        print('-' * 30)
                                    print('[bos] ' + ll)



                            else:
                                label_tag = inputs['sst_codes']
                                # score_lst = result_dict['score_lst']
                                for ii, ll in enumerate(inputs['sents']):
                                    if ii % int(len(inputs['sents']) / len(self.discri_labels)) == 0:
                                        print('-' * 30)

                                    # print(label_tag[ii])
                                    # print(gen_code[ii])
                                    print('{},{} | {},{} |[bos] '.format(self.discri_labels[gen_code[ii]],
                                                                         -1,
                                                                         self.discri_labels[label_tag[ii]],
                                                                         -1,) + ll)



                inputs = self.cleanup_samples(inputs)
                gen_code = inputs['sst_codes']


                tr_loss_temp, result_dict = self.training_step_dataless(model, inputs,
                                                                        self.discri_model, train_buffer=False)
                tr_loss += tr_loss_temp

                if self.replay_buffer:
                    # save the current examples to the buffer.
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.cpu()
                    # print(result_dict)
                    self.buffer_lst.append((inputs, result_dict))

                    # sampling from previous buffer to train.
                    for (inputs_s, result_dict_s) in random.sample(self.buffer_lst, k=min(5, len(self.buffer_lst[:-1]))):
                        tr_loss_temp, _ = self.training_step_dataless(model, (inputs_s, result_dict_s),
                                                                      self.discri_model, train_buffer=True)
                        tr_loss += tr_loss_temp




                # if verbose and step % 10 == 9:
                #     with torch.no_grad():
                #         # print(inputs['input_ids'])
                #         # print(inputs['labels'])
                #         if self.sample_from_gpt:
                #             print('sampling from q distribution for test; in training loop, we sample from GPT2.')
                #             inputs_ = self.get_dataless_input(attribute_type=self.dataless_control_type,
                #                                              sample_size=self.dataless_sample_size,
                #                                              sample_from_gpt=False)
                #             # TODO.
                #
                #             inputs_['sents'] = [self.tokenizer.decode(ll[1:]) for ll in inputs_['input_ids'].tolist()]
                #
                #             for ii, ll in enumerate(inputs_['sents']):
                #                 if ii % int(len(inputs['sents'])/len(self.discri_labels) ) == 0:
                #                     print('-'*30)
                #                 print('[bos] ' + ll)
                #
                #
                #
                #         else:
                #             label_tag = inputs['sst_codes']
                #             score_lst = result_dict['score_lst']
                #             for ii, ll in enumerate(inputs['sents']):
                #                 if ii % int(len(inputs['sents']) / len(self.discri_labels)) == 0:
                #                     print('-'*30)
                #
                #                 # print(label_tag[ii])
                #                 # print(gen_code[ii])
                #                 print('{},{} | {},{} |[bos] '.format(self.discri_labels[gen_code[ii]], score_lst[ii][gen_code[ii]],
                #                                                      self.discri_labels[label_tag[ii]], score_lst[ii][label_tag[ii]])  + ll)



                # print()
                # # DEBUG
                # for param in list(model.parameters())[-5:]:
                #     print(param.requires_grad, end='')
                #     print(param.mean(), end=' ')
                # print()

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    num_update_steps_per_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == num_update_steps_per_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / num_update_steps_per_epoch

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if (
                        self.args.evaluation_strategy == EvaluationStrategy.STEPS
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        if self.hp_search_backend is not None and trial is not None:
                            run_id = (
                                trial.number
                                if self.hp_search_backend == HPSearchBackend.OPTUNA
                                else tune.get_trial_id()
                            )
                            checkpoint_folder += f"-run-{run_id}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.store_flos()

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                metrics = self.evaluate()
                self._report_to_hp_search(trial, epoch, metrics)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)


    def update_gumbel_temp(self, curr_step, total_step):
        final_temp = 0.5
        init_temp = INIT_GUMBEL_TEMP
        self.gumbel_temperature = init_temp - (init_temp - final_temp)/total_step * curr_step


    # URGENT
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def ids_to_clean_text(self, tokenizer, generated_ids):
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                # print(model, model.module)
                if self.args.n_gpu > 1:
                    self.total_flos = getattr(model.module.config, "total_flos", 0)
                else:
                    self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                #URGENT
                # aa_sent = self.ids_to_clean_text(self.tokenizer, inputs['input_ids'])
                # for aa in aa_sent:
                #     print(aa)
                tr_loss += self.training_step(model, inputs)
                # URGENT
                # print('total_loss', tr_loss)

                if False and self.task_mode == 'keyword' and (step) % 5000 == 0:
                    with torch.no_grad():
                        # prompting with [BOS]
                        encoded_prompt = self.tokenizer.encode('[BOS] ', add_special_tokens=False,
                                                          return_tensors="pt")
                        encoded_prompt = encoded_prompt.to(self.model.device)
                        inputs_ = self.get_dataless_input(attribute_type=5,
                                                         sample_size=10,
                                                         sample_from_gpt=False, input_ids_prompt=encoded_prompt)

                        inputs_['sents'] = [self.tokenizer.decode(ll[1:]) for ll in inputs_['input_ids'].tolist()]

                        for ii, ll in enumerate(inputs_['sents']):
                            # if ii % int(len(inputs_['sents']) / len(self.discri_labels)) == 0:
                            #     print('-' * 30)
                            print('[bos] ' + ll)
                elif False and self.task_mode == 'keyword' and (step) % 5000 == 0:
                    with torch.no_grad():
                        # prompting with [BOS]
                        encoded_prompt = self.tokenizer.encode('[BOS] ', add_special_tokens=False,
                                                          return_tensors="pt")
                        encoded_prompt = encoded_prompt.to(self.model.device)
                        inputs_ = self.get_dataless_input(attribute_type=3,
                                                         sample_size=10,
                                                         sample_from_gpt=False, input_ids_prompt=encoded_prompt)

                        inputs_['sents'] = [self.tokenizer.decode(ll[1:]) for ll in inputs_['input_ids'].tolist()]

                        for ii, ll in enumerate(inputs_['sents']):
                            # if ii % int(len(inputs_['sents']) / len(self.discri_labels)) == 0:
                            #     print('-' * 30)
                            print('[bos] ' + ll)





                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # URGENT
                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)


                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    # print(self.args.evaluation_strategy == EvaluationStrategy.STEPS )
                    # print(self.global_step % self.args.eval_steps == 0)
                    # print()

                    if (
                        self.args.evaluation_strategy == EvaluationStrategy.STEPS
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)

                        # URGENT, this is just for the low data setting, False otherwise.
                        #####################################################
                        if 'lowdata' in self.args.output_dir or 'earlystop' in self.args.output_dir:
                            self.save_based_on_eval = True
                        else:
                            self.save_based_on_eval = False
                        print('if not see a line lowdata: below, then did not go into low data. ')
                        if self.save_based_on_eval and metrics["eval_loss"] < self.curr_best_eval:
                            print('lowdata:', self.global_step, self.curr_best_eval, metrics["eval_loss"],
                                  'perplexity={}'.format(math.exp(metrics["eval_loss"])))
                            self.curr_best_eval = metrics["eval_loss"]
                            if hasattr(model, "module"):
                                assert (
                                        model.module is self.model
                                ), f"Module {model.module} should be a reference to self.model"
                            else:
                                assert model is self.model, f"Model {model} should be a reference to self.model"
                            # Save model checkpoint
                            output_dir_name = os.path.basename(self.args.output_dir)
                            checkpoint_folder = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                            if self.hp_search_backend is not None and trial is not None:
                                run_id = (
                                    trial.number
                                    if self.hp_search_backend == HPSearchBackend.OPTUNA
                                    else tune.get_trial_id()
                                )
                                checkpoint_folder += f"-run-{run_id}"
                            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                            self.store_flos()
                            print('saving to output_dir', output_dir)
                            self.save_model(output_dir)

                            if self.is_world_process_zero():
                                self._rotate_checkpoints(use_mtime=True)
                        #####################################################

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        print('saving model at a checkpoint!!')
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        if self.hp_search_backend is not None and trial is not None:
                            run_id = (
                                trial.number
                                if self.hp_search_backend == HPSearchBackend.OPTUNA
                                else tune.get_trial_id()
                            )
                            checkpoint_folder += f"-run-{run_id}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.store_flos()

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                metrics = self.evaluate()
                self._report_to_hp_search(trial, epoch, metrics)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        **kwargs
    ) -> BestRun:
        """
        Launch an hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objectie`, which defaults to a function returning the evaluation loss when no metric is provided,
        the sum of all metrics otherwise.

        .. warning::

            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

        Args:
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.trainer_utils.default_hp_space_optuna` or
                :func:`~transformers.trainer_utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:

                - the documentation of `optuna.create_study <https://optuna.readthedocs.io/en/stable/reference/alias_generated/optuna.create_study.html#optuna.create_study>`__
                - the documentation of `tune.run <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__

        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the informations about the best run.
        """
        if backend is None:
            backend = default_hp_search_backend()
            if backend is None:
                raise RuntimeError(
                    "At least one of optuna or ray should be installed. "
                    "To install optuna run `pip install optuna`."
                    "To install ray run `pip install ray[tune]`."
                )
        backend = HPSearchBackend(backend)
        if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
            raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
        if backend == HPSearchBackend.RAY and not is_ray_available():
            raise RuntimeError(
                "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
            )
        self.hp_search_backend = backend

        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        run_hp_search = run_hp_search_optuna if backend == HPSearchBackend.OPTUNA else run_hp_search_ray
        best_run = run_hp_search(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        # Set up loggers like W&B or Comet ML
        self._setup_loggers()

        if hasattr(self, "_log"):
            warnings.warn(
                "The `_log` method is deprecated and won't be called in a future version, define `log` in your subclass.",
                FutureWarning,
            )
            return self._log(logs, iterator=iterator)

        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                logs["total_flos"] = self.total_flos
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_process_zero():
                wandb.log(logs, step=self.global_step)
        if is_comet_available():
            if self.is_world_process_zero():
                experiment = comet_ml.config.get_global_experiment()
                if experiment is not None:
                    experiment._log_metrics(logs, step=self.global_step, epoch=self.epoch, framework="transformers")
        output = {**logs, **{"step": self.global_step}}
        if self.is_world_process_zero():
            self.log_history.append(output)
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            assert  False
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        if self.use_dropout:
            if self.gpt2 is not None:
                self.gpt2.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                if self.distill:
                    loss = self.compute_loss_distill(model, inputs, gpt2_model=self.gpt2, )
                else:
                    loss = self.compute_loss(model, inputs, gpt2_model=self.gpt2)
        else:
            if self.distill:
                loss = self.compute_loss_distill(model, inputs, gpt2_model=self.gpt2)
            else:
                loss = self.compute_loss(model, inputs, gpt2_model=self.gpt2)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # print(loss)
            loss.backward()

        # print('max allocated_memory:', torch.cuda.max_memory_allocated(0), 'total_memory:', torch.cuda.get_device_properties(0).total_memory,
        #       'percentage', torch.cuda.max_memory_allocated(0)/torch.cuda.get_device_properties(0).total_memory)


        return loss.detach()

    def training_step_dataless(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                               discri_model:Optional[PreTrainedModel]=None,
                               train_buffer:Optional[bool] = False) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        if self.use_dropout:
            if self.gpt2 is not None:
                self.gpt2.train()
        # TODO.
        if not train_buffer:
            inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                if self.adaptive_data:
                    loss, result_dict = self.compute_loss_ADAP(model, inputs, gpt2_model=self.gpt2, train_buffer=train_buffer)
                else:
                    loss, result_dict = self.compute_loss_REINFORCE(model, inputs, gpt2_model=self.gpt2, train_buffer=train_buffer)
        else:
            if self.adaptive_data:
                loss, result_dict = self.compute_loss_ADAP(model, inputs, gpt2_model=self.gpt2, train_buffer=train_buffer)
            else:
                loss, result_dict = self.compute_loss_REINFORCE(model, inputs, gpt2_model=self.gpt2, train_buffer=train_buffer)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # print(loss)
            loss.backward()

        return loss.detach(), result_dict

    def convert_tokens(self, src_tokenizer, tgt_tokenizer):
        src_v = src_tokenizer.vocab_size
        tgt_v = tgt_tokenizer.vocab_size
        src_s = src_tokenizer.convert_ids_to_tokens(list(range(src_v)))
        # print(src_s)
        tgt_vocab = tgt_tokenizer.get_vocab()
        count = 0
        for l in src_s[:200]:
            if l not in tgt_vocab:
                print(l, end=' ||| ')
                count += 1
        print(count, 200, src_v, tgt_v)

        return

    def get_length_logits(self, input_sent, len_criterion):
        len_lst = []
        for sent in input_sent:
            tok_sent = word_tokenize(sent)
            len_lst.append(len(tok_sent))
        len_lst = torch.tensor(len_lst).float()
        # print(len_lst, len_criterion, len_lst.shape)
        len_criterion_ = len_criterion.unsqueeze(0).expand(len(input_sent), -1)

        # len_criterion_ = len_criterion.unsqueeze(1).expand(-1, self.dataless_sample_size).reshape(-1)
        # print(len_lst.shape, len_criterion_.shape)
        diff = torch.abs(len_lst.unsqueeze(1).expand(-1, len(len_criterion)) - len_criterion_).to(self.args.device)
        # print(diff.shape)
        logits = torch.log_softmax(-diff, dim=1)
        # print(logits)
        # print(logits.exp())
        return len_lst, logits

    def compute_loss_REINFORCE(self, model, inputs, gpt2_model=None, train_buffer=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        device = model.device

        if train_buffer:
            if self.forward_kl:
                # print('training based on replay buffer')
                reward = inputs[1]['imp_w'].to(device)
                inputs_ = inputs[0]
                inputs_ = self._prepare_inputs(inputs_)
                q_output = self.model(input_ids=inputs_['input_ids'], labels=inputs_['labels'],
                                      control_code=inputs_['control_code'], gpt2_model=gpt2_model)  # q(x|a)
                q_logp = -q_output.loss
                # print(reward)
                # print(q_logp)
                loss = -(reward * q_logp).mean()
                # print(loss)
                # print('end buffer')
                return loss, None
            else:
                assert False, 'currently only forward kl can use experience replay buffer. '

        result_dict = {}

        sst_codes = inputs['sst_codes']
        gpt2_output = gpt2_model(input_ids=inputs['input_ids'], labels=inputs['labels'],
                                 output_hidden_states=True)
        gpt2_logp = -gpt2_output.loss
        # print(inputs['input_ids'].shape, inputs['control_code'].shape)
        q_output = self.model(input_ids=inputs['input_ids'], labels=inputs['labels'],
                              control_code=inputs['control_code'], gpt2_model=gpt2_model) # q(x|a)
        q_logp = -q_output.loss
        # print(gpt2_logp-q_logp)

        if self.dataless_control_type == 0:
            discri_output = self.discri_model(torch.mean(gpt2_output.hidden_states[-1], dim=1))
            discri_output = torch.softmax(discri_output, dim=1)
            pos = discri_output.index_select(1, torch.LongTensor([0, 2]).to(device)).sum(1)
            neg = discri_output.index_select(1, torch.LongTensor([1, 3]).to(device)).sum(1)
            neu = discri_output.index_select(1, torch.LongTensor([4]).to(device)).sum(1)
            score_lst = torch.stack([pos, neg, neu], dim=1).log()
            discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        elif self.dataless_control_type == 2:
            discri_input = self.discri_tokenizer(inputs['sents'], return_tensors="pt", padding=True,
                                  is_split_into_words=False, return_offsets_mapping=False, add_special_tokens=True)
            discri_input = self._prepare_inputs(discri_input)
            # discri_input['labels'] = inputs['sst_codes']
            discri_output = self.discri_model(**discri_input)
            score_lst = discri_output.logits
            score_lst = torch.log_softmax(score_lst, dim=1)
            # print(score_lst.shape, sst_codes.shape)
            discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        elif self.dataless_control_type == 3:
            # controling for the length of the data.
            # print(self.discri_model)
            len_lst, score_lst = self.get_length_logits(inputs['sents'], self.discri_model)
            # print(sst_codes)
            discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
            print(discri_logp.exp())



        # print(gpt2_logp.shape, discri_logp.shape, q_logp.shape)
        # print(gpt2_logp[:3])
        # print(discri_logp[:3])
        # print(q_logp[:3])

        if self.gumbel:
            sample_size = 1
            q_logits = q_output.logits #bsz,seqlen,vocab
            q_logp_t = torch.log_softmax(q_logits, dim=-1)
            q_prob_t = q_logp_t.exp()

            p_logits = gpt2_output.logits  # bsz,seqlen,vocab
            p_logp_t = torch.log_softmax(p_logits, dim=-1)
            p_prob_t = p_logp_t.exp()

            if False:
                bsz, seqlen, vocab_size = q_logits.shape
                unif = torch.rand(bsz, sample_size, seqlen, vocab_size).to(device)
                gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
                log_alpha = q_logits.unsqueeze(1).expand(-1, sample_size, -1, -1)
                logit = (log_alpha + gumbel) / self.gumbel_temperature
                q_sample_t = torch.softmax(logit, dim=-1)

                print(q_sample_t.shape)
                q_sample_t = q_sample_t.squeeze(1)
                print(q_sample_t.shape)
                p_logits = gpt2_output.logits #bsz,seqlen,vocab
                print(p_logits.shape, q_logits.shape)
                p_prob_t = torch.softmax(p_logits, dim=-1)
                p_logp_t = p_prob_t.log()

            entropy = -(q_prob_t * q_logp_t).sum(dim=-1)
            # print(entropy.shape,)
            reward1 = (q_prob_t * p_prob_t).sum(dim=-1).log()
            reward2 = (q_prob_t * p_logp_t).sum(dim=-1)
            # print(reward1.shape, reward1.mean(), reward2.mean())

            # print(discri_logp.shape, q_logp.shape)
            reward = (reward1 + entropy).mean(dim=1) + discri_logp * q_logp
            # print(reward.shape)
            result_dict['imp_w'] = reward.detach().exp().data.cpu()
            result_dict['gpt2_logp'] = gpt2_logp.data.cpu()
            result_dict['discri_logp'] = discri_logp.data.cpu()
            result_dict['q_logp'] = q_logp.data.cpu()
            # result_dict['loss'] = loss.item()
            result_dict['score_lst'] = score_lst.data.cpu()

            return -(reward).mean(), result_dict
            # reward = (q_sample_t * p_logp_t).sum(dim=-1) + discri_score_t
            # loss = -reward.mean()

            # NOT SURE ABOUT HOW TO BREAK THIS DOWN TO PREFIX REWARD.
            # discri_input_ids = discri_input['input_ids']
            # discri_embed_weights = self.discri_model.base_model.get_input_embeddings()
            # discri_hard_input = discri_embed_weights(discri_input_ids)
            # print(q_sample_t.shape, discri_embed_weights.weight.shape)
            # # IMPORTANT: need to align between word embeddings.
            # self.convert_tokens(self.tokenizer, self.discri_tokenizer)
            # discri_soft_input = torch.bmm(q_sample_t, discri_embed_weights.weight)   # bsz, seqlen, embs
            # print(discri_input)
            # # mix and match with the prefix.
            # print(discri_soft_input.shape, discri_hard_input.shape)
            # discri_hard_input = discri_hard_input.split(1, dim=1)
            # discri_soft_input = discri_soft_input.split(1, dim=1)

            # for i in range(seqlen):
            #     temp_lst = discri_hard_input[:i] + discri_hard_input[i]
            #     discri_embs_input = torch.cat(temp_lst, dim=1)
            #     print(discri_embs_input.shape, discri_input_ids.shape, )
            #     discri_output_t = self.discri_model(inputs_embeds=discri_embs_input, output_hidden_states=True,
            #                                         attention_mask=None)
            #     score_lst = torch.log_softmax(discri_output_t.logits, dim=1)
            #     discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
            #     discri_score_t.append(discri_logp)



        elif self.reverse_kl and self.forward_kl:
            pass

        elif self.reverse_kl:
            # reward = gpt2_logp + discri_logp - q_logp
            reward = gpt2_logp + discri_logp - 0.5*q_logp
            if True:
                if self.dataless_usebaseline:
                    reward_ = reward.view(-1, self.dataless_sample_size)
                    baseline = torch.zeros_like(reward_)
                    baseline_k = torch.zeros_like(reward_) # bsz
                    for k in range(self.dataless_sample_size):
                        baseline_k.copy_(reward_)
                        baseline_k[:, k].fill_(0)
                        baseline[:, k] = baseline_k.detach().sum(1) / (self.dataless_sample_size - 1)
                    # print('see ', baseline.detach(), result['state_llq'].mean())
                    # print(reward_.shape, baseline.shape, q_logp.shape)
                    reinforce = (reward_.detach() - baseline.detach()).view(-1) *  q_logp
                else:
                    reinforce = reward.detach() * q_logp
                    # print('here')
            else:
                reinforce = torch.zeros(reward.shape).to(device)

            print(reward[:3], reinforce[:3])
            print(gpt2_logp[:3])
            print(discri_logp[:3])
            print(q_logp[:3])

            elbo = -(reward.mean() + reinforce.mean())
            loss = elbo
            print(loss)

        elif self.forward_kl:

            if self.sample_from_gpt:
                reward = discri_logp
            else:
                reward = gpt2_logp + discri_logp - q_logp
            reward2 = reward.exp().detach() * q_logp
            # print(reward[:3], reward2[:3])
            # print(gpt2_logp[:3])
            # print(discri_logp[:3])
            # print(q_logp[:3])
            loss = -(reward2.mean())
            # print(loss)
            result_dict['imp_w'] = reward.detach().exp().data.cpu()
            result_dict['gpt2_logp'] = gpt2_logp.data.cpu()
            result_dict['discri_logp'] = discri_logp.data.cpu()
            result_dict['q_logp'] = q_logp.data.cpu()
            result_dict['loss'] = loss.item()
            result_dict['score_lst'] = score_lst.data.cpu()


        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]


        return loss, result_dict


    def compute_loss_ADAP(self, model, inputs, gpt2_model=None, train_buffer=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        device = model.device

        if train_buffer:
            if self.forward_kl:
                # print('training based on replay buffer')
                reward = inputs[1]['imp_w'].to(device)
                inputs_ = inputs[0]
                inputs_ = self._prepare_inputs(inputs_)
                q_output = self.model(input_ids=inputs_['input_ids'], labels=inputs_['labels'],
                                      control_code=inputs_['control_code'], gpt2_model=gpt2_model)  # q(x|a)
                q_logp = -q_output.loss
                # print(reward)
                # print(q_logp)
                loss = -(reward * q_logp).mean()
                # print(loss)
                # print('end buffer')
                return loss, None
            else:
                assert False, 'currently only forward kl can use experience replay buffer. '

        result_dict = {}
        # adaptively re-adapt the labels for the samples.
        if self.dataless_control_type == 0:
            discri_output = self.discri_model(torch.mean(gpt2_output.hidden_states[-1], dim=1))
            discri_output = torch.softmax(discri_output, dim=1)
            pos = discri_output.index_select(1, torch.LongTensor([0, 2]).to(device)).sum(1)
            neg = discri_output.index_select(1, torch.LongTensor([1, 3]).to(device)).sum(1)
            neu = discri_output.index_select(1, torch.LongTensor([4]).to(device)).sum(1)
            score_lst = torch.stack([pos, neg, neu], dim=1).log()
            discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        elif self.dataless_control_type == 2:
            discri_input = self.discri_tokenizer(inputs['sents'], return_tensors="pt", padding=True,
                                  is_split_into_words=False, return_offsets_mapping=False, add_special_tokens=True)
            discri_input = self._prepare_inputs(discri_input)
            # discri_input['labels'] = inputs['sst_codes']
            discri_output = self.discri_model(**discri_input)
            score_lst = discri_output.logits
            score_lst = torch.log_softmax(score_lst, dim=1)
            max_score = torch.max(score_lst, dim=1)
            discri_logp = max_score.values
            # print(discri_logp.shape, discri_logp)
            k = torch.cat(self.discri_labels_code, dim=0)
            control_code = torch.index_select(k, 0, max_score.indices)
            inputs['sst_codes'] = max_score.indices
            inputs['control_code'] = control_code
            # print(control_code)
            # print()
            # print(k)
            # print(max_score.indices)
            # print(score_lst.shape, sst_codes.shape)
            # discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        elif self.dataless_control_type == 3:
            # controling for the length of the data.
            # print(self.discri_model)
            len_lst, score_lst = self.get_length_logits(inputs['sents'], self.discri_model)
            # print(sst_codes)
            discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
            print(discri_logp.exp())

        gpt2_output = gpt2_model(input_ids=inputs['input_ids'], labels=inputs['labels'],
                                 output_hidden_states=True)
        gpt2_logp = -gpt2_output.loss
        q_output = self.model(input_ids=inputs['input_ids'], labels=inputs['labels'],
                              control_code=inputs['control_code'], gpt2_model=gpt2_model) # q(x|a)
        q_logp = -q_output.loss
        # print(gpt2_logp-q_logp)


        assert self.forward_kl


        if self.sample_from_gpt:
            reward = discri_logp
        else:
            reward = gpt2_logp + discri_logp - q_logp
        reward2 = reward.exp().detach() * q_logp
        # print(reward[:3], reward2[:3])
        # print(gpt2_logp[:3])
        # print(discri_logp[:3])
        # print(q_logp[:3])
        loss = -(reward2.mean())
        # print(loss)
        result_dict['imp_w'] = reward.detach().exp().data.cpu()
        result_dict['gpt2_logp'] = gpt2_logp.data.cpu()
        result_dict['discri_logp'] = discri_logp.data.cpu()
        result_dict['q_logp'] = q_logp.data.cpu()
        result_dict['loss'] = loss.item()
        result_dict['score_lst'] = score_lst.data.cpu()


        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # print(loss, result_dict)
        return loss, result_dict

    def compute_loss(self, model, inputs, gpt2_model=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # outputs = model.forward_weighted(**inputs)
        if 'prompt_lab' in inputs:
            prompt_lab_ = inputs['prompt_lab']
            k = torch.cat(self.discri_labels_code, dim=0)
            inputs['control_code'] = torch.index_select(k, 0, prompt_lab_)
            del inputs['prompt_lab']

        outputs = model(**inputs, gpt2_model=gpt2_model)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # print(outputs[0])
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # print(outputs[0], outputs.loss)
        # URGENT
        # print('compute_loss', outputs[0])
        return outputs[0].mean()

    def compute_loss_distill(self, model, inputs, gpt2_model=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # outputs = model.forward_weighted(**inputs)

        with torch.no_grad():
            output_finetuned = self.finetuned_gpt2(**inputs)

        outputs = model(**inputs, gpt2_model=gpt2_model)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.matching_objective == 'kl':
            # distrib_finetuned=torch.log_softmax(output_finetuned.logits[:,:,:-2], dim=-1)  #bsz, seqlen, vocab
            distrib_finetuned=torch.log_softmax(output_finetuned.logits, dim=-1)  #bsz, seqlen, vocab
            distrib_prefix = torch.log_softmax(outputs.logits, dim=-1)  # bsz, seqlen, vocab
            loss = torch.sum(distrib_finetuned.exp() * (distrib_finetuned - distrib_prefix), dim=-1) #bsz, seqlen

        elif self.matching_objective == 'logits':
            loss = torch.norm(output_finetuned.logits - outputs.logits, dim=-1)  #bsz, seqlen
            # loss = torch.norm(output_finetuned.logits[:,:,:-2] - outputs.logits, dim=-1)  #bsz, seqlen

        elif self.matching_objective == 'last_layer':
            activation_diff = output_finetuned.last_hidden_state - outputs.last_hidden_state
            loss = torch.norm(activation_diff, dim=-1)  # bsz, seqlen
        else:
            assert False, "invalid matching_objective"

        return  loss.sum(dim=-1).mean()

    def is_local_master(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.

        .. warning::

            This method is deprecated, use :meth:`~transformers.Trainer.is_local_process_zero` instead.
        """
        warnings.warn("This method is deprecated, use `Trainer.is_local_process_zero()` instead.", FutureWarning)
        return self.is_local_process_zero()

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on
        several machines, this is only going to be :obj:`True` for one process).

        .. warning::

            This method is deprecated, use :meth:`~transformers.Trainer.is_world_process_zero` instead.
        """
        warnings.warn("This method is deprecated, use `Trainer.is_world_process_zero()` instead.", FutureWarning)
        return self.is_world_process_zero()

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on
        several machines, this is only going to be :obj:`True` for one process).
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_process_zero():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            json.dump(
                self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=2, ensure_ascii=False
            )

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        json.dump(
            self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=2, ensure_ascii=False
        )

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                self.model.config.total_flos = total_flos

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        output_dir_name = os.path.basename(self.args.output_dir)
        checkpoint_prefix = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}"

        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics



    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed.

        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self.prediction_loop(test_dataloader, description="Prediction")

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        entropy_losses: List[float] = []
        model.eval()
        if self.gpt2 is not None:
            self.gpt2.eval()

        print(model.training)
        print(self.gpt2.training)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)
            if logits is not None:
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
                temp_logits = [torch.log_softmax(x) for x in logits]
                entropy_losses.extend([(x.exp() * x).sum() for x in temp_logits])
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = nested_xla_mesh_reduce(preds, "eval_preds")
            if label_ids is not None:
                label_ids = nested_xla_mesh_reduce(label_ids, "eval_label_ids")
            if eval_losses is not None:
                eval_losses = xm.mesh_reduce("eval_losses", torch.tensor(eval_losses), torch.cat).tolist()

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                    .mean()
                    .item()
                )
            else:
                metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        if len(entropy_losses) > 0:
            metrics['entropy'] = np.mean(entropy_losses)
            print('entropy', metrics['entropy'] )

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        # At eval time, set the weights to 1/bsz. and see the results..

        # if 'weights' in inputs:
        #     weights = inputs['weights']
        #     bsz = weights.view(-1).shape[0]
        #     weights = (torch.ones(weights.shape)/bsz).to(weights.device)
        #     inputs['weights'] = weights

        with torch.no_grad():
            # outputs = model.forward_weighted(**inputs)
            outputs = model(**inputs, gpt2_model=self.gpt2)
            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from :class:`~transformers.PretrainedModel`, uses
        that method to compute the number of floating point operations for every backward + forward pass. If using
        another model, either implement such a method in the model or subclass and override this method.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """

        if isinstance(self.model, torch.nn.DataParallel) or isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, "floating_point_ops"):
            return model.floating_point_ops(inputs)

        else:
            return 0

    def gen_data(self, data_path: Optional[str] = None, total_steps:Optional[int] = 7250, #250K
                       verbose: Optional[bool] = False):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.

        file_o = open(data_path, 'w')

        with torch.no_grad():
            for step in range(total_steps):
                inputs = self.get_dataless_input(attribute_type=self.dataless_control_type,
                                                 sample_size=self.dataless_sample_size,
                                                 sample_from_gpt=True)
                # remove the trailing eos tokens.
                input_text = [self.tokenizer.decode(ll[1:], clean_up_tokenization_spaces=True) for ll in
                              inputs['input_ids'].tolist()]
                inputs['sents'] = []
                to_del = []
                for idx_, text in  enumerate(input_text):
                    text_idx = text.find(self.tokenizer.eos_token)
                    if text_idx < 0:
                        text_idx2 = text.rfind('.')
                        text_idx2 = max(text_idx2, text.rfind('!'))
                        text_idx2 = max(text_idx2, text.rfind('?'))
                        if text_idx2 < 0:
                            inputs['sents'].append(text)
                            to_del.append(idx_)
                        else:
                            inputs['sents'].append(text[:text_idx2+1])
                    else:
                        inputs['sents'].append(text[:text_idx])
                # print(inputs['sents'])
                inputs = self._prepare_inputs(inputs)
                tr_loss_temp, result_dict = self.compute_loss_REINFORCE(self.model, inputs, gpt2_model=self.gpt2, train_buffer=False)

                gpt2_logp = result_dict['gpt2_logp']
                score_lst = result_dict['score_lst']
                print(gpt2_logp.shape, score_lst.shape, len(to_del))
                for idx, line in enumerate(inputs['sents']):

                    if idx in to_del:
                        # print(line)
                        continue
                    '''
                    result_dict['imp_w'] = reward.detach().exp().data.cpu()
                    result_dict['gpt2_logp'] = gpt2_logp.data.cpu()
                    result_dict['discri_logp'] = discri_logp.data.cpu()
                    result_dict['q_logp'] = q_logp.data.cpu()
                    result_dict['loss'] = loss.item()
            
                    '''
                    file_o.write('[bos] {}|||{}|||{}\n'.format(line, gpt2_logp[idx], score_lst[idx].tolist()))

        file_o.close()


    def train_amortized_pplm(self, model_path: Optional[str] = None,
                       trial: Union["optuna.Trial", Dict[str, Any]] = None,
                       verbose: Optional[bool] = False):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        # train_dataloader = self.get_train_dataloader()
        # num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        # num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        # num_update_steps_per_epoch = 5
        num_update_steps_per_epoch = 15000
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                # print(model, model.module)
                if self.args.n_gpu > 1:
                    self.total_flos = getattr(model.module.config, "total_flos", 0)
                else:
                    self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):

            #

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(range(num_update_steps_per_epoch), desc="Iteration", disable=disable_tqdm)
            # for step, inputs in enumerate(epoch_iterator):
            for step in range(num_update_steps_per_epoch):

                # Skip past any already trained steps if resuming training
                # if steps_trained_in_current_epoch > 0:
                #     steps_trained_in_current_epoch -= 1
                #     epoch_pbar.update(1)
                #     continue

                with torch.no_grad():
                    inputs = self.get_dataless_input(attribute_type=self.dataless_control_type,
                                                     sample_size=self.dataless_sample_size,
                                                     sample_from_gpt=self.sample_from_gpt)


                # inputs = self.cleanup_samples(inputs)
                gen_code = inputs['sst_codes']


                tr_loss_temp, result_dict = self.training_step_amortized_pplm(model, inputs,
                                                                        self.discri_model, train_buffer=False)
                tr_loss += tr_loss_temp

                ########################################################################
                if verbose and step % 20 == 9:
                    with torch.no_grad():
                        # print(inputs['input_ids'])
                        # print(inputs['labels'])
                        if self.sample_from_gpt:
                            print('sampling from q distribution for test; in training loop, we sample from GPT2.')
                            inputs_ = self.get_dataless_input(attribute_type=self.dataless_control_type,
                                                             sample_size=self.dataless_sample_size,
                                                             sample_from_gpt=False)
                            # TODO.

                            inputs_['sents'] = [self.tokenizer.decode(ll[1:]) for ll in inputs_['input_ids'].tolist()]

                            for ii, ll in enumerate(inputs_['sents']):
                                if ii % int(len(inputs['sents'])/len(self.discri_labels) ) == 0:
                                    print('-'*30)
                                print('[bos] ' + ll)



                        else:
                            inputs['sents'] = [self.tokenizer.decode(ll[1:]) for ll in inputs['input_ids'].tolist()]
                            label_tag = inputs['sst_codes']
                            score_lst = result_dict['score_lst']
                            print(gen_code, label_tag, score_lst)
                            for ii, ll in enumerate(inputs['sents']):
                                if ii % int(len(inputs['sents']) / len(self.discri_labels)) == 0:
                                    print('-'*30)

                                # print(label_tag[ii])
                                # print(gen_code[ii])
                                # print('{},{} | {},{} |[bos] '.format(self.discri_labels[gen_code[ii]], score_lst[ii][gen_code[ii]],
                                #                                      self.discri_labels[label_tag[ii]], score_lst[ii][label_tag[ii]])  + ll)
                                print('[bos] {}'.format(ll))

                ########################################################################


                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    num_update_steps_per_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == num_update_steps_per_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / num_update_steps_per_epoch

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if (
                        self.args.evaluation_strategy == EvaluationStrategy.STEPS
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        if self.hp_search_backend is not None and trial is not None:
                            run_id = (
                                trial.number
                                if self.hp_search_backend == HPSearchBackend.OPTUNA
                                else tune.get_trial_id()
                            )
                            checkpoint_folder += f"-run-{run_id}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.store_flos()

                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                metrics = self.evaluate()
                self._report_to_hp_search(trial, epoch, metrics)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)


    def training_step_amortized_pplm(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                                       discri_model:Optional[PreTrainedModel]=None,
                                       train_buffer:Optional[bool] = False) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        if self.use_dropout:
            if self.gpt2 is not None:
                self.gpt2.train()
        # TODO.
        if not train_buffer:
            inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss, result_dict = self.compute_loss_amortized_pplm(model, inputs, gpt2_model=self.gpt2)

        else:
            loss, result_dict = self.compute_loss_amortized_pplm(model, inputs, gpt2_model=self.gpt2)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # print(loss)
            loss.backward()

        return loss.detach(), result_dict

    def compute_loss_amortized_pplm(self, model, inputs, gpt2_model=None, discri_model=None,):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        device = model.device


        result_dict = {}

        # regular model forward.
        bsz, seqlen = inputs['input_ids'].shape
        # print(inputs['control_code'], inputs['sst_codes'], inputs['sst_codes'].shape, inputs['control_code'].shape)
        model_output = model(input_ids=inputs['input_ids'], output_hidden_states=True, use_cache=True,
                             control_code=inputs['control_code'], gpt2_model=gpt2_model)
        model_logp = torch.log_softmax(model_output.logits, dim=-1)

        # compute the KL divergence with the language model terms.
        gpt2_output = gpt2_model(input_ids=inputs['input_ids'], output_hidden_states=True, use_cache=True)
        gpt2_logp = torch.log_softmax(gpt2_output.logits, dim=-1)

        # say this is mode seeking KL(q||p)
        kl_loss = (model_logp.exp() * (model_logp - gpt2_logp)).sum(dim=-1) # bsz, seqlen
        # print(kl_loss.shape)

        # Get the loss from the discriminative classifier.
        # gumbel softmax trick.
        gumbel_softmax = False
        if gumbel_softmax:
            soft_input_states = None  # bsz, seqlen, vocab
        else:
            soft_input_states = model_logp.exp() #bsz, seqlen, vocab

        discri_embed_weights = self.gpt2.base_model.get_input_embeddings()
        # print(soft_input_states.shape, discri_embed_weights.weight.shape)
        embedding_input = torch.bmm(soft_input_states, discri_embed_weights.weight.unsqueeze(0).expand(bsz, -1, -1))  #bsz, seqlen, embs
        embedding_input = torch.split(embedding_input, 1, dim=1) # a list of length seqlen; each element: (bsz, embs)
        # print(len(embedding_input), embedding_input[0].shape)
        # compute the hidden states:
        gpt2_past = gpt2_output.past_key_values
        gpt2_past = torch.stack(gpt2_past, dim=0) # layer, 2, batch_size, num_heads, sequence_length, embed_size_per_head
        gpt2_hidden = gpt2_output.hidden_states[-1] # bsz, seqlen, hidden
        discri_loss = []
        for i in range(0, seqlen):

            gpt2_temp_past = torch.narrow(gpt2_past, 4, 0, i)
            gpt2_temp_hidden = torch.narrow(gpt2_hidden, 1, 0, i)
            # print(gpt2_temp_past.shape, gpt2_temp_hidden.shape)
            temp_out = gpt2_model(inputs_embeds=embedding_input[i], past_key_values=gpt2_temp_past,
                                  output_hidden_states=True)
            # print(temp_out.hidden_states[-1].shape, len(temp_out.hidden_states))
            # print(temp_out.hidden_states[-1][0])

            temp_out_accum = torch.cat([gpt2_temp_hidden, temp_out.hidden_states[-1]], dim=1)
            # print(temp_out_accum.shape)
            discri_output = self.discri_model(torch.mean(temp_out_accum, dim=1))
            discri_output = torch.softmax(discri_output, dim=1)
            if True:
                pos = discri_output.index_select(1, torch.LongTensor([2]).to(device)).sum(1)
                neg = discri_output.index_select(1, torch.LongTensor([3]).to(device)).sum(1)
            else:
                pos = discri_output.index_select(1, torch.LongTensor([0, 2]).to(device)).sum(1)
                neg = discri_output.index_select(1, torch.LongTensor([1, 3]).to(device)).sum(1)
            # neu = discri_output.index_select(1, torch.LongTensor([4]).to(device)).sum(1)
            score_lst = torch.stack([pos, neg], dim=1).log()
            # print(score_lst.shape)
            discri_logp = score_lst.gather(1, inputs['sst_codes'].unsqueeze(-1)).sum(1)  # bsz, 1
            # print(score_lst.exp())
            # print()
            # print(discri_logp.exp(),)
            discri_loss.append(discri_logp)
        # print(discri_loss)
        discri_logp = torch.stack(discri_loss, dim=1) # bsz, seqlen
        # print(discri_logp.shape, kl_loss.shape)
        # print(discri_logp[0], kl_loss[0])
        loss = (-discri_logp + kl_loss).mean()

        result_dict['loss'] = loss.item()
        result_dict['discri_logp'] = discri_logp.data.cpu()
        result_dict['kl_loss'] = kl_loss.data.cpu()
        result_dict['score_lst'] = discri_logp.mean(dim=1).data.cpu()
        # result_dict['gpt2_logp'] = gpt2_logp.data.cpu()
        # result_dict['model_logp'] = model_logp.data.cpu()
        # result_dict['score_lst'] = score_lst.data.cpu()

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # print(loss, result_dict)
        return loss, result_dict




        #
        # if self.pplm_control_type == 0:
        #     discri_output = self.discri_model(torch.mean(gpt2_output.hidden_states[-1], dim=1))
        #     discri_output = torch.softmax(discri_output, dim=1)
        #     pos = discri_output.index_select(1, torch.LongTensor([0, 2]).to(device)).sum(1)
        #     neg = discri_output.index_select(1, torch.LongTensor([1, 3]).to(device)).sum(1)
        #     neu = discri_output.index_select(1, torch.LongTensor([4]).to(device)).sum(1)
        #     score_lst = torch.stack([pos, neg, neu], dim=1).log()
        #     discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        # elif self.dataless_control_type == 2:
        #     discri_input = self.discri_tokenizer(inputs['sents'], return_tensors="pt", padding=True,
        #                           is_split_into_words=False, return_offsets_mapping=False, add_special_tokens=True)
        #     discri_input = self._prepare_inputs(discri_input)
        #     # discri_input['labels'] = inputs['sst_codes']
        #     discri_output = self.discri_model(**discri_input)
        #     score_lst = discri_output.logits
        #     score_lst = torch.log_softmax(score_lst, dim=1)
        #     max_score = torch.max(score_lst, dim=1)
        #     discri_logp = max_score.values
        #     # print(discri_logp.shape, discri_logp)
        #     k = torch.cat(self.discri_labels_code, dim=0)
        #     control_code = torch.index_select(k, 0, max_score.indices)
        #     inputs['sst_codes'] = max_score.indices
        #     inputs['control_code'] = control_code
        #
        # elif self.dataless_control_type == 3:
        #     # controling for the length of the data.
        #     # print(self.discri_model)
        #     len_lst, score_lst = self.get_length_logits(inputs['sents'], self.discri_model)
        #     # print(sst_codes)
        #     discri_logp = score_lst.gather(1, sst_codes.unsqueeze(-1)).sum(1)  # bsz, 1
        #     print(discri_logp.exp())


        ###################################################
        # q_output = self.model(input_ids=inputs['input_ids'], labels=inputs['labels'],
        #                       control_code=inputs['control_code'], gpt2_model=gpt2_model) # q(x|a)
        # q_logp = -q_output.loss
        # print(gpt2_logp-q_logp)


        # assert self.forward_kl


        # if self.sample_from_gpt:
        #     reward = discri_logp
        # else:
        #     reward = gpt2_logp + discri_logp - q_logp
        # reward2 = reward.exp().detach() * q_logp
        # # print(reward[:3], reward2[:3])
        # # print(gpt2_logp[:3])
        # # print(discri_logp[:3])
        # # print(q_logp[:3])
        # loss = -(reward2.mean())
        # print(loss)






