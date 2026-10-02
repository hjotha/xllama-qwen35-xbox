// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/llama_ini.h"

#include <doctest/doctest.h>

#include <string>

using namespace xllama;

TEST_CASE("llama_ini: basic key=value") {
    const LlamaIni m = parse_llama_ini("n_gpu_layers=99\nkv_q8=1\n");
    CHECK(m.at("n_gpu_layers") == "99");
    CHECK(m.at("kv_q8") == "1");
}

TEST_CASE("llama_ini: comments, section headers and blank lines") {
    const LlamaIni m = parse_llama_ini("# full-line comment\n"
                                       "; another comment\n"
                                       "\n"
                                       "[llama]\n"
                                       "n_ctx = 4096\n"
                                       "\n"
                                       "[other]\n"
                                       "n_threads = 8\n");
    // Headers are ignored: one flat namespace.
    CHECK(m.at("n_ctx") == "4096");
    CHECK(m.at("n_threads") == "8");
}

TEST_CASE("llama_ini: whitespace and case are forgiven, last wins") {
    const LlamaIni m = parse_llama_ini("  N_CTX = 2048 \nN_ctx=4096\n");
    CHECK(m.size() == 1);
    CHECK(m.at("n_ctx") == "4096");
}

TEST_CASE("llama_ini: empty and malformed input") {
    CHECK(parse_llama_ini("").empty());
    CHECK(parse_llama_ini("# only a comment\n").empty());
    CHECK(parse_llama_ini("no-equals-here\n").empty());
    CHECK(parse_llama_ini("=novalue\n").empty());
    const LlamaIni m = parse_llama_ini("n_batch=\n");
    CHECK(m.at("n_batch") == "");
}

TEST_CASE("llama_ini_int: strict whole-value integers") {
    const LlamaIni m = parse_llama_ini("a=42\nb=12x\nc=\nd=-7\n");
    int v = 0;
    CHECK(llama_ini_int(m, "a", v));
    CHECK(v == 42);
    CHECK_FALSE(llama_ini_int(m, "b", v));
    CHECK_FALSE(llama_ini_int(m, "c", v));
    CHECK_FALSE(llama_ini_int(m, "missing", v));
    CHECK(llama_ini_int(m, "d", v));
    CHECK(v == -7);
}

TEST_CASE("llama_ini_bool: 1/0 and words") {
    const LlamaIni m = parse_llama_ini("a=1\nb=0\nc=TRUE\nd=Off\ne=maybe\n");
    bool v = false;
    CHECK(llama_ini_bool(m, "a", v));
    CHECK(v);
    CHECK(llama_ini_bool(m, "b", v));
    CHECK_FALSE(v);
    CHECK(llama_ini_bool(m, "c", v));
    CHECK(v);
    CHECK(llama_ini_bool(m, "d", v));
    CHECK_FALSE(v);
    CHECK_FALSE(llama_ini_bool(m, "e", v));
    CHECK_FALSE(llama_ini_bool(m, "missing", v));
}
