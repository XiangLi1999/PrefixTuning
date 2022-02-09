#!/bin/sh

# init silkflow
silkflow=/mnt/nas/public/release/silkflow_group/silkflow_framework/lastest/silkflow_framework/tools/silkflow
if [ -e $silkflow ]; then
    echo "link silkflow ..."
    ln -sf $silkflow /bin/silkflow
    ln -sf $silkflow /usr/bin/silkflow
    echo "done."
fi

# install libs package
echo "install libs package ..."
cd /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers;pip install -i https://mirrors.aliyun.com/pypi/simple -e .;pip install -i https://mirrors.aliyun.com/pypi/simple nltk
status=$?
if [ $status -ne 0 ]; then
    echo -e "\n----------------------------------------------------"
    echo "execute cd /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers;pip install -i https://mirrors.aliyun.com/pypi/simple -e .;pip install -i https://mirrors.aliyun.com/pypi/simple nltk failed!"
    exit $status
fi
echo "done."

# check and add user
id -u jieyixin.jyx >/dev/null 2>&1
status=$?
if [ $status -ne 0 ]; then
    useradd -m -s /bin/bash -N -u 1371220 jieyixin.jyx
    status=$?
    if [ $status -ne 0 ]; then
        echo -e "\n----------------------------------------------------"
        echo "transformers-20220114-13_07_31 switch to user jieyixin.jyx failed!"
        exit $status
    fi
fi

echo "export PYTHONPATH=/mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers:/mnt/nas/public/release/silkflow_group/silkflow_framework/lastest:/mnt/nas/public/release/silkflow_group/silkflow_mt/lastest:/mnt/nas/public/release/silkflow_group/silkflow_nlp/lastest" >> /etc/profile
# switch user
su - jieyixin.jyx <<'EOF'
# init conda env
if [ -d "" ]; then
    if [ ! -d "" ]; then
        mkdir -p 
    fi
    echo "link  ..."
    ln -sf  
    echo "init conda config ..."
    CONDA_PATH=
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_PATH/bin:$PATH"
    fi
    if [ -n "" ]; then
        echo "activate conda env# ..."
        conda activate 
        status=$?
        if [ $status -ne 0 ]; then
            echo -e "\n----------------------------------------------------"
            echo "conda activate  failed!"
            exit $status
        fi
    fi
    echo "done."
fi
echo "start transformers-20220114-13_07_31 ..."
echo -e "----------------------------------------------------\n"
# started
touch /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/started
echo "transformers-20220114-13_07_31 started" >> /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/run_status
# cd to experiment dir
cd /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers
# execute command
sh -c "bash /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/gpt2/nct_models/nctprefixtune_y_5_act_cat_b\=5-e\=5_d\=0.0_u\=no_lr\=5e-05_w\=0.0_s\=101_r\=n_m\=512_o\=1_o\=1.sh" 2>&1 | tee /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/log
status=${PIPESTATUS[0]}
echo "execute status:$status"
if [ $status -ne 0 ]; then
    echo -e "\n----------------------------------------------------"
    echo "transformers-20220114-13_07_31 failed"
    touch /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/failed
    echo "transformers-20220114-13_07_31 failed" >> /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/run_status
    rm -f /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/started
    exit $status
fi
echo -e "\n----------------------------------------------------"
echo "transformers-20220114-13_07_31 success"
# create checkpoint file
touch /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/done
rm -f /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/failed
rm -f /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/transformers-20220114-13_07_31/started
echo "transformers-20220114-13_07_31 done" >> /mnt/nas/users/jieyixin.jyx/workspace/gitlab.alibaba-inc.com/jieyixin.jyx/runGPT2/Kexin/PrefixTuning/transformers/silkflow_detail/meta/run_status
exit $?
EOF