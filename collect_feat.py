# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import torch
import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [[]]
    # dialogs = []
    system_msg = "Please follow the rest pattern to generate the answer with single or multiple entity words for the [token] " \
                 "place in the pattern sentence." \
                 "The first pattern is:  The [mainpart] contains [token]. For example, 'The dog contains legs, torse, fur, head, tail'" \
                 # "The second pattern is: 'The [subpart] belongs to [token]'."

    feat_collection = {
        'token_ids': [],
        'tokens': [],
        'feats': []
    }

    chat_id = 0
    dialogs[chat_id].append({"role": "system", "content": f"{system_msg}"})
    chat_id = -1
    STOP = False
    while not STOP:
        # if len(dialogs) > 0:
        #     dialogs[-1] = []
        # dialogs.append([])
        # chat_id = chat_id + 1
        # dialogs[-1].append({"role": "system", "content": f"{system_msg}"})

        mainpart = 'cat'

        prompt = f"The {mainpart} contains [token]"
        # prompt = "Please provide a hierarchical description for appearance of a dog including its detailed sub-part name. Notes that only the visible parts included and the inner part like skull and its sub-part should be excluded."

        dialogs[chat_id].append(
            {
                "role": "user", "content": f"{prompt}"
            }
                                )
        results = generator.chat_completion_feat(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        result = f"{results[chat_id]['generation']['role'].capitalize()}: {results[chat_id]['generation']['content']}"
        print(f"{result}\n======\n")

        # test = [(t, f) for t, f in zip(results[0]['generation']['tokens'], results[0]['generation']['corr_feats']) if
        #         'dog' in t]
        # corr = torch.stack([_[1] for _ in test])
        # corr_norm = corr / corr.norm(dim=-1, keepdim=True)
        # print(corr_norm @ corr_norm.transpose(1, 0))

        generation = {"role": f"{results[chat_id]['generation']['role']}",
                      "content": f"{results[chat_id]['generation']['content']}"}
        dialogs[chat_id].append(generation)


if __name__ == "__main__":
    fire.Fire(main)
