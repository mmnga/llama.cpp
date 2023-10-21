import argparse
import json
import sys
import os
import torch
import numpy as np
from pathlib import Path
import gguf
from sentencepiece import SentencePieceProcessor  # type: ignore[import]

try:
    from safetensors import safe_open
except ImportError:
    print("Please install `safetensors` python package")
    sys.exit(1)


def count_model_parts(dir_model: Path) -> int:
    # get number of model parts
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("model-00"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PLaMo model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "ftype", type=int, choices=[0, 1], default=1, nargs='?',
        help="output format - use 0 for float32, 1 for float16",
    )
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

if hparams["architectures"][0] != "PlamoForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit(1)

# get number of model parts
num_parts = count_model_parts(dir_model)

# from add PLaMo model #3557
# https://github.com/ggerganov/llama.cpp/pull/3557/files

ARCH=gguf.MODEL_ARCH.PLAMO
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_hidden_layers"]

gguf_writer.add_name("PLaMo")
gguf_writer.add_context_length(4096) # not in config.json
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_head_count(hparams["num_attention_heads"])
gguf_writer.add_head_count_kv(hparams["num_attention_heads"] // hparams["n_shared_head"])
gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
gguf_writer.add_file_type(ftype)


# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytes] = []
scores: list[float] = []
toktypes: list[int] = []

tokenizer_model_file = dir_model / 'tokenizer.model'
if not tokenizer_model_file.is_file():
    print(f'Error: Missing {tokenizer_model_file}', file = sys.stderr)
    sys.exit(1)

# vocab type sentencepiece
print("gguf: get sentencepiece tokenizer vocab, scores and token types")

tokenizer = SentencePieceProcessor(str(tokenizer_model_file))

for i in range(tokenizer.vocab_size()):
    text: bytes
    score: float

    piece = tokenizer.id_to_piece(i)
    text = piece.encode("utf-8")
    score = tokenizer.get_score(i)

    toktype = 1  # defualt to normal token type
    if tokenizer.is_unknown(i):
        toktype = 2
    if tokenizer.is_control(i):
        toktype = 3

    # toktype = 4 is user-defined = tokens from added_tokens.json

    if tokenizer.is_unused(i):
        toktype = 5
    if tokenizer.is_byte(i):
        toktype = 6

    tokens.append(text)
    scores.append(score)
    toktypes.append(toktype)

gguf_writer.add_tokenizer_model("llama")
gguf_writer.add_token_list(tokens)
gguf_writer.add_token_scores(scores)
gguf_writer.add_token_types(toktypes)
gguf_writer.add_sep_token_id(5)
gguf_writer.add_pad_token_id(3)

special_vocab = gguf.SpecialVocab(dir_model)
special_vocab.add_to_gguf(gguf_writer)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# params for qkv transform
n_head    = hparams["num_attention_heads"]
n_head_kv = hparams["num_key_value_heads"]

head_dim = hparams["hidden_size"] // n_head

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = iter(("model.safetensors",))
else:
    part_names = (
        f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    model_part = safe_open(dir_model / part_name, framework="pt")

    for name in model_part.keys():
        if "self_attn.rotary_emb.inv_freq" in name:
            continue
        data = model_part.get_tensor(name)

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
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

        print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(new_name, data)


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
