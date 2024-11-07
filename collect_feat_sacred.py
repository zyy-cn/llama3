# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama import Llama

from sacred import Experiment
from easydict import EasyDict as edict
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ex = Experiment('llama3')

def create_basic_stream_logger(format):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
ex.logger = create_basic_stream_logger('%(levelname)s - %(name)s - %(message)s')
ex.add_config('./configs/llama3.yaml')


@ex.capture
def chat_completion(cfg, generator, dialogs, _log):

    temperature = cfg.model.temperature
    top_p = cfg.model.top_p
    max_gen_len = None if 0 == cfg.model.max_gen_len else cfg.model.max_gen_len

    chat_id = -1

    mainpart = 'cat'

    prompt = f"The {mainpart} contains [token]"

    dialogs[chat_id].append(
        {
            "role": "user", "content": f"{prompt}"
        }
                            )
    dialog = f"{dialogs[chat_id][-1]['role'].capitalize()}: {dialogs[chat_id][-1]['content']}"

    _log.info(dialog)

    results = generator.chat_completion_feat(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    result = f"{results[chat_id]['generation']['role'].capitalize()}: {results[chat_id]['generation']['content']}"
    _log.info(f"{result}\n======\n")

    # test = [(t, f) for t, f in zip(results[0]['generation']['tokens'], results[0]['generation']['corr_feats']) if
    #         'dog' in t]
    # corr = torch.stack([_[1] for _ in test])
    # corr_norm = corr / corr.norm(dim=-1, keepdim=True)
    # print(corr_norm @ corr_norm.transpose(1, 0))

    generation = {"role": f"{results[chat_id]['generation']['role']}",
                  "content": f"{results[chat_id]['generation']['content']}"}
    dialogs[chat_id].append(generation)

    return dialogs


@ex.automain
def main(_run, _log):

    cfg = edict(_run.config)

    ckpt_dir = cfg.model.ckpt_dir
    tokenizer_path = cfg.model.tokenizer_path
    max_seq_len = cfg.model.max_seq_len
    max_batch_size = cfg.model.max_batch_size


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [[]]
    system_msg = "Please follow the rest pattern to generate the answer with single or multiple entity words for the [token] " \
                 "place in the pattern sentence." \
                 "The first pattern is:  The [mainpart] contains [token]. For example, 'The dog contains legs, torse, fur, head, tail'" \

    chat_id = 0
    STOP = False
    dialogs[chat_id].append({"role": "system", "content": f"{system_msg}"})
    dialog = f"{dialogs[chat_id][-1]['role'].capitalize()}: {dialogs[chat_id][-1]['content']}"
    _log.info(dialog)
    while not STOP:
        dialogs = chat_completion(cfg, generator, dialogs)
        STOP = True