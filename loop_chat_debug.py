# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

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

    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    # results = generator.chat_completion(
    #     dialogs,  # type: ignore
    #     max_gen_len=max_gen_len,
    #     temperature=temperature,
    #     top_p=top_p,
    # )
    dialogs = [[]]
    prompt = ""
    chat_id = 0
    # system_msg = input(f"System {chat_id}:")
    system_msg = ""
    if system_msg != "":
        dialogs[chat_id].append({"role": "system", "content": f"{system_msg}"})
    while prompt != 'exit':
        # prompt = input(f"======\nuser {chat_id}:")
        # prompt = "please generate detailed descriptions for the following categories: dog"
        # prompt = "Please provide a comprehensive description of dog including its appearance, features, characteristics, functions, and any relevant information."
        prompt = "Please provide a hierarchical description for appearance of a dog including its detailed sub-part name. Notes that only the visible parts included and the inner part like skull and its sub-part should be excluded."
        dialogs[chat_id].append({"role": "user", "content": f"{prompt}"})
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        result = f"{results[chat_id]['generation']['role'].capitalize()}: {results[chat_id]['generation']['content']}"
        print(f"{result}\n======\n")
        generation = {"role": f"{results[chat_id]['generation']['role']}",
                      "content": f"{results[chat_id]['generation']['content']}"}
        dialogs[chat_id].append(generation)
        break


if __name__ == "__main__":
    fire.Fire(main)
