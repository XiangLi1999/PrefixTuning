# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from typing import List
from silkflow_framework.sdk.base_pipeline import BasePipeline
from silkflow_framework.sdk.expt_manager import ExptManager


class PipelineTemplate(BasePipeline):
    """A pipeline template to run train"""

    def __init__(self, expt_manager: ExptManager, prefix: str, dep_ops: List = [], configs: dict = {},
                 params: dict = {}):
        if not prefix:
            prefix = self.__class__.__name__
        default_params = {}
        # update experiment directory.
        silkflow_detail_dir = '%s/silkflow_detail' % (os.getcwd() if expt_manager is None else expt_manager.expt_dir)
        if not expt_manager or not hasattr(expt_manager, '_update_runtime_status'):
            expt_manager = ExptManager(expt_dir=silkflow_detail_dir)
        else:
            expt_manager._update_runtime_status(expt_dir=silkflow_detail_dir)
        # init base pipeline.
        super()._init(expt_manager, prefix, dep_ops=dep_ops, configs=configs, params=params,
                      default_params=default_params)

    def _define(self):
        action_op = super()._add_op(name="transformers-20220114-13_47_12",
                                    image="reg.docker.alibaba-inc.com/silkflow/pytorch:1.8.1-cuda11.1-cudnn8-devel",
                                    command="cd /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers;bash /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/gpt2/nct_models/nctprefixtune_y_5_act_cat_b\=5-e\=5_d\=0.0_u\=no_lr\=5e-05_w\=0.0_s\=101_r\=n_m\=512_o\=1_o\=1.sh",
                                    gpus=2,
                                    cpu=4,
                                    memory=16,
                                    requirements="cd /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers;pip install -i https://mirrors.aliyun.com/pypi/simple -e .;pip install -i https://mirrors.aliyun.com/pypi/simple nltk;pip install -i https://mirrors.aliyun.com/pypi/simple tokenizers==0.8.1.rc2",
                                    node_selector={})
        self.last_ops += action_op