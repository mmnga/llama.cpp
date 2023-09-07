#ifndef SPM_TOKENIZER
#define SPM_TOKENIZER

#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <map>
#include <unordered_map>
#include <queue>
#include <cstring>
#include <regex>

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_SPM = 0, // SentencePiece
    LLAMA_VOCAB_TYPE_BPE = 1, // Byte Pair Encoding
};

typedef int llama_token;

enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};

struct llama_vocab {
    // TODO:
    // - add a vector of merges
    //   so that we can pass it to different types of tokenizers with a common interface

    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    id linefeed_id = 13;
};

//
// helpers
//

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

//
// tokenizer
//

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_CONTROL;
}

static bool llama_is_user_defined_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_USER_DEFINED;
}

static bool llama_is_unused_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNUSED;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_BYTE;
}

static bool llama_is_bos_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(llama_is_control_token(vocab, id));
    return id == vocab.special_bos_id;
}

static bool llama_is_eos_token(const llama_vocab & vocab, llama_token id ) {
    GGML_ASSERT(llama_is_control_token(vocab, id));
    return id == vocab.special_eos_id;
}

static bool llama_is_pad_token(const llama_vocab & vocab, llama_token id ) {
    GGML_ASSERT(id < 0 || llama_is_control_token(vocab, id));
    return id == vocab.special_pad_id;
}

static uint8_t llama_token_to_byte(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto& token_data = vocab.id_to_token.at(id);
    auto buf = token_data.text.substr(3, 2);
    return strtol(buf.c_str(), NULL, 16);
}

static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch) {
    char buf[7];
    int result = snprintf(buf, sizeof(buf), "<0x%02X>", ch);
    GGML_ASSERT(0 <= result && result < 7);
    return vocab.token_to_id.at(buf);
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}
static std::string get_llama_escape_whitespac(const std::string& text) {
    std::string result = "\xe2\x96\x81";
    for (size_t offs = 0; offs < text.length(); ++offs) {
        if (text[offs] == ' ') {
            result += "\xe2\x96\x81";
        } else {
            result += text[offs];
        }
    }
    return result;
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = utf8_len(text[offs]);
            GGML_ASSERT(offs + len <= text.size());
            sym.text = text.c_str() + offs;
            sym.n = len;
            offs += len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

// static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
//     std::vector<char> result(8, 0);
//     const int n_tokens = llama_token_to_piece(ctx, token, result.data(), result.size());
//     if (n_tokens < 0) {
//         result.resize(-n_tokens);
//         int check = llama_token_to_piece(ctx, token, result.data(), result.size());
//         GGML_ASSERT(check == -n_tokens);
//     } else {
//         result.resize(n_tokens);
//     }

//     return std::string(result.data(), result.size());
// }

// does not write null-terminator to str
int llama_token_to_str(const llama_vocab & vocab, llama_token token, char * buf, int length) {
    if (llama_is_normal_token(vocab, token)) {
        std::string result = vocab.id_to_token[token].text;
        if (llama_vocab_get_type(vocab) == LLAMA_VOCAB_TYPE_SPM) {
            llama_unescape_whitespace(result);
        }
        if (length < (int) result.length()) {
            return -result.length();
        }
        memcpy(buf, result.c_str(), result.length());
        return result.length();
    } else if (llama_is_unknown_token(vocab, token)) { // NOLINT
        if (length < 3) {
            return -3;
        }
        buf[0] = '\xe2';
        buf[1] = '\x96';
        buf[2] = '\x85';
        return 3;
    } else if (llama_is_control_token(vocab, token)) {
        ;
    } else if (llama_is_byte_token(vocab, token)) {
        if (length < 1) {
            return -1;
        }
        buf[0] = llama_token_to_byte(vocab, token);
        return 1;
    }
    return 0;
}

// display token
static std::string llama_token_to_text(const llama_vocab & vocab, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_str(vocab, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_str(vocab, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

// int llama_token_to_piece(const struct llama_context * ctx, llama_token token, char * buf, int length) {
//     return llama_token_to_piece_with_model(&ctx->model, token, buf, length);
// }

// // does not write null-terminator to buf
// int llama_token_to_piece_with_model(const struct llama_model * model, llama_token token, char * buf, int length) {
//     if (0 <= token && token < llama_model_n_vocab(model)) {
//         if (llama_is_normal_token(model->vocab, token)) {
//             std::string result = model->vocab.id_to_token[token].text;
//             if (llama_vocab_get_type(model->vocab) == LLAMA_VOCAB_TYPE_SPM) {
//                 llama_unescape_whitespace(result);
//             }
//             if (length < (int) result.length()) {
//                 return -result.length();
//             }
//             memcpy(buf, result.c_str(), result.length());
//             return result.length();
//         } else if (llama_is_unknown_token(model->vocab, token)) { // NOLINT
//             if (length < 3) {
//                 return -3;
//             }
//             buf[0] = '\xe2';
//             buf[1] = '\x96';
//             buf[2] = '\x85';
//             return 3;
//         } else if (llama_is_control_token(model->vocab, token)) {
//             ;
//         } else if (llama_is_byte_token(model->vocab, token)) {
//             if (length < 1) {
//                 return -1;
//             }
//             buf[0] = llama_token_to_byte(model->vocab, token);
//             return 1;
//         }
//     }
//     return 0;
// }

static std::vector<llama_vocab::id> spm_tokenize(const llama_vocab & vocab, const std::string & raw_text, bool bos, bool escape) {
    std::vector<llama_vocab::id> output;

    if (raw_text.empty()) {
        return output;
    }

    switch (vocab.type) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                llm_tokenizer_spm tokenizer(vocab);

                if (bos) {
                    output.push_back(vocab.special_bos_id);
                }

                std::string text;
                if (escape) {
                    text = get_llama_escape_whitespac(raw_text);
                } else {
                    text = raw_text;
                }
                tokenizer.tokenize(text, output);
            } break;
        default:
            {
                GGML_ASSERT(false);
            }
    };

    return output;
}

#endif // SPM_TOKENIZER
