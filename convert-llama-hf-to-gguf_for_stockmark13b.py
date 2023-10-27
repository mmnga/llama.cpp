# HF llama --> gguf conversion

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore[import]
from safetensors import safe_open

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf

# reverse HF permute back to original pth layout
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py


def reverse_hf_permute(weights: NDArray, n_head: int, n_kv_head: Optional[int] = None) -> NDArray:
    if n_kv_head is not None and n_head != n_kv_head:
        n_head //= n_kv_head

    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def count_model_parts(dir_model: str) -> int:
    num_parts = 0

    for filename in os.listdir(dir_model):
        if filename.startswith("model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")

    return num_parts


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")

    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a GPT-J model to a GGML compatible file")
    parser.add_argument("--vocab-only",  action="store_true",    help="extract only the vocab")
    parser.add_argument("--outfile",     type=Path,              help="path to write to; default: based on input")
    parser.add_argument("model",         type=Path,              help="directory containing model file, or model file itself (*.bin)")
    parser.add_argument("ftype",     type=int, choices=[0, 1],   help="output format - use 0 for float32, 1 for float16", default = 1)
    return parser.parse_args()

args = parse_args()

dir_model = args.model
ftype = args.ftype
if not dir_model.is_dir():
    print(f'Error: {args.model} is not a directory', file = sys.stderr)
    sys.exit(1)

# map from ftype to string
ftype_str = ["f32", "f16"]

if args.outfile is not None:
    fname_out = args.outfile
else:
    # output in the same directory as the model by default
    fname_out = dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'

print("gguf: loading model "+dir_model.name)

with open(dir_model / "config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

if hparams["architectures"][0] != "LlamaForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.LLAMA
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_hidden_layers"]
head_count = hparams["num_attention_heads"]

if "num_key_value_heads" in hparams:
    head_count_kv = hparams["num_key_value_heads"]
else:
    head_count_kv = head_count

if "_name_or_path" in hparams:
    hf_repo = hparams["_name_or_path"]
else:
    hf_repo = ""

if "max_sequence_length" in hparams:
    ctx_length = hparams["max_sequence_length"]
elif "max_position_embeddings" in hparams:
    ctx_length = hparams["max_position_embeddings"]
else:
    print("gguf: can not find ctx length parameter.")

    sys.exit()

gguf_writer.add_name(dir_model.name)
gguf_writer.add_source_hf_repo(hf_repo)
gguf_writer.add_tensor_data_layout("Meta AI original pth")
gguf_writer.add_context_length(ctx_length)
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
gguf_writer.add_rope_dimension_count(hparams["hidden_size"] // hparams["num_attention_heads"])
gguf_writer.add_head_count(head_count)
gguf_writer.add_head_count_kv(head_count_kv)
gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

if "rope_scaling" in hparams and hparams["rope_scaling"] != None and "factor" in hparams["rope_scaling"]:
    if "type" in hparams["rope_scaling"]:
        if hparams["rope_scaling"]["type"] == "linear":
            gguf_writer.add_rope_scale_linear(hparams["rope_scaling"]["factor"])


# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: List[bytes] = []
scores: List[float] = []
toktypes: List[int] = []
tokenizer_json_file = dir_model / 'tokenizer.json'

#if Path(dir_model + "/tokenizer.model").is_file():
if tokenizer_json_file.is_file():
    # gpt2 tokenizer
    gguf_writer.add_tokenizer_model("gpt2")

    with open(tokenizer_json_file, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    print("gguf: get gpt2 tokenizer vocab")

    vocab_size = len(tokenizer_json["model"]["vocab"])

    # ref: https://github.com/cmp-nct/ggllm.cpp/blob/master/falcon_convert.py
    tokenizer = AutoTokenizer.from_pretrained(dir_model, use_fast=True)
    vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
    assert max(tokenizer.vocab.values()) < vocab_size

    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
    for i in range(vocab_size):
        tokens.append(reverse_vocab[i] if i in reverse_vocab else f"[PAD{i}]")
        scores.append(0.0) # dummy
        toktypes.append(gguf.TokenType.NORMAL)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges = True)
    special_vocab.add_to_gguf(gguf_writer)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = ("model.safetensors",)
else:
    part_names = (
        f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    ctx = safe_open(dir_model / part_name, framework="pt", device="cpu")
    with ctx as model_part:
        for name in model_part.keys():
            data = model_part.get_tensor(name)

            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            data = data.squeeze().numpy()

            # reverse permute these
            if name.endswith(".q_proj.weight"):
                data = reverse_hf_permute(data, head_count)
            if name.endswith(".k_proj.weight"):
                data = reverse_hf_permute(data, head_count, head_count_kv)

            # map tensor names
            if name.endswith(".weight") and name[:-7] in tensor_map:
                name = tensor_map[name[:-7]] + ".weight"
            elif name.endswith(".bias") and name[:-5] in tensor_map:
                name = tensor_map[name[:-5]] + ".bias"
            else:
                print("Can not map tensor '" + name + "'")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

            gguf_writer.add_tensor(name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
print("gguf: write tensors")
gguf_writer.write_tensors_to_file()

gguf_writer.close()


print(f"gguf: model successfully exported to '{fname_out}'")
print("")
