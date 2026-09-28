#include "dflash-gpu-draft-sha256.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    const auto path = std::filesystem::temp_directory_path() /
                      "dflash-draft-sha256-abc.bin";
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write("abc", 3);
        if (!file) return 1;
    }
    std::string hex, error;
    const bool ok = dflash_draft_sha256_file(path.string(), &hex, &error);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    if (!ok || hex !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        std::fprintf(stderr, "SHA-256 abc mismatch: %s %s\n",
                     hex.c_str(), error.c_str());
        return 2;
    }
    if (argc > 1) {
        if (!dflash_draft_sha256_file(argv[1], &hex, &error)) {
            std::fprintf(stderr, "SHA-256 file failed: %s\n", error.c_str());
            return 3;
        }
        std::printf("%s\n", hex.c_str());
    } else {
        std::puts("SHA-256 abc: PASS");
    }
    return 0;
}
