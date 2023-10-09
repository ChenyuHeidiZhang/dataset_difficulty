nlprun \
    -a rlhf -o /nlp/scr/chenyuz/dataset_difficulty/dwmw_checks.log \
    --mail-user chenyuz@stanford.edu -g 2 -m sphinx1 -r 100G \
    'accelerate launch data_checklist.py'


# accelerate config
# /nlp/u/chenyuz/.cache/huggingface/accelerate/default_config.yaml