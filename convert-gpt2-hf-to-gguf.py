# HF gptneox--> gguf conversion

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

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def count_model_parts(dir_model: str) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


def count_model_parts(dir_model: Path) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


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

# possible tensor data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16

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

if hparams["architectures"][0] != "GPT2LMHeadModel":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.GPT2
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["n_layer"]


gguf_writer.add_name(dir_model.name)
gguf_writer.add_context_length(hparams["n_positions"])
gguf_writer.add_embedding_length(hparams["n_embd"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_head_count(hparams["n_head"])
gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytearray] = []
toktypes: list[int] = []
scores: list[float] = []

tokenizer_json_file = dir_model / 'tokenizer.json'
# 
# bpe tokenizer
# 
if tokenizer_json_file.is_file():

    # gpt2 tokenizer
    gguf_writer.add_tokenizer_model("gpt2")

    with open(tokenizer_json_file, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    print("gguf: get gpt2 tokenizer vocab")

    vocab_size = len(tokenizer_json["model"]["vocab"])

    # ref: https://github.com/cmp-nct/ggllm.cpp/blob/master/falcon_convert.py
    tokenizer = AutoTokenizer.from_pretrained(dir_model, use_fast=True)

    reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    for i in range(vocab_size):
        if i in reverse_vocab:
            try:
                text = bytearray([byte_decoder[c] for c in reverse_vocab[i]])
            except KeyError:
                text = bytearray()
                for c in reverse_vocab[i]:
                    if ord(c) < 256:  # single byte character
                        text.append(byte_decoder[ord(c)])
                    else:  # multibyte special token character
                        text.extend(c.encode('utf-8'))
        else:
            print(f"Key {i} not in tokenizer vocabulary. Padding with an arbitrary token.")
            pad_token = f"[PAD{i}]".encode("utf8")
            text = bytearray(pad_token)

        tokens.append(text)
        toktypes.append(gguf.TokenType.NORMAL)  # dummy

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(dir_model, load_merges = True)
    special_vocab.add_to_gguf(gguf_writer)

# 
# spm tokenizer
# 
else:

    # spm tokenizer
    gguf_writer.add_tokenizer_model("spm")

    print("gguf: get spm tokenizer vocab")

    # added tokens
    added_tokens_json_file = dir_model / 'added_tokens.json'
    added_tokens_json = {}
    if added_tokens_json_file.is_file():
        with open(added_tokens_json_file, "r", encoding="utf-8") as f:
            added_tokens_json = json.load(f)

    reverse_added_tokens = { added_tokens_json[key]:key for key in added_tokens_json}

    sp_tokenizer = AutoTokenizer.from_pretrained(dir_model, use_fast=False)
    sp_vocab = sp_tokenizer.get_vocab()
    vocab_size = len(sp_vocab)

    for i in range(vocab_size):
        text = ""
        toktype = gguf.TokenType.NORMAL  # defualt to normal token type
        score = 0.0 # dummy

        try:
            piece = sp_tokenizer.sp_model.id_to_piece(i)
            text = piece.encode("utf-8")
            score = sp_tokenizer.sp_model.get_score(i)

            if sp_tokenizer.sp_model.is_unknown(i):
                toktype = gguf.TokenType.UNKNOWN
            if sp_tokenizer.sp_model.is_control(i):
                toktype = gguf.TokenType.CONTROL
            if sp_tokenizer.sp_model.is_unused(i):
                toktype = gguf.TokenType.UNUSED
            if sp_tokenizer.sp_model.is_byte(i):
                toktype = gguf.TokenType.BYTE
        except:
            # added tokens piece id is out of range.
            toktype = gguf.TokenType.USER_DEFINED
            score = 0.0
            if i in reverse_added_tokens:
                text = reverse_added_tokens[i].encode("utf-8")
                score = 0.0

        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
            
    if sp_tokenizer.bos_token_id is not None :
        print("add bos token ",sp_tokenizer.bos_token_id, tokens[sp_tokenizer.bos_token_id])
        gguf_writer.add_bos_token_id(sp_tokenizer.bos_token_id)

    if sp_tokenizer.eos_token_id is not None :
        print("add eos token ",sp_tokenizer.eos_token_id, tokens[sp_tokenizer.eos_token_id])
        gguf_writer.add_eos_token_id(sp_tokenizer.eos_token_id)
    
    if sp_tokenizer.unk_token_id is not None :
        print("add unk token ",sp_tokenizer.unk_token_id, tokens[sp_tokenizer.unk_token_id])
        gguf_writer.add_unk_token_id(sp_tokenizer.unk_token_id)
    
    if sp_tokenizer.sep_token_id is not None :
        print("add sep token ",sp_tokenizer.sep_token_id, tokens[sp_tokenizer.sep_token_id])
        gguf_writer.add_sep_token_id(sp_tokenizer.sep_token_id)
    
    if sp_tokenizer.pad_token_id is not None :
        print("add pad token ",sp_tokenizer.pad_token_id, tokens[sp_tokenizer.pad_token_id])
        gguf_writer.add_pad_token_id(sp_tokenizer.pad_token_id)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = iter(("pytorch_model.bin",))
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")

    for name in model_part.keys():
        data = model_part[name]

        print(name)

        # we don't need these
        if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
            continue

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

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
            
        if ftype == 1 and name.endswith("pos_embd.weight"):
            data = data.astype(np.float32)

        # for efficiency - transpose these matrices:
        # Transpose ATTN_QKV, ATTN_OUT, FFN_UP, FFN_DOWN
        if name.endswith(".attn_qkv.weight") \
        or name.endswith(".attn_output.weight") \
        or name.endswith(".ffn_up.weight") \
        or name.endswith(".ffn_down.weight"):
            print("Transposing ", name, data.shape)
            data = data.transpose()

        print(name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
if not args.vocab_only:
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
