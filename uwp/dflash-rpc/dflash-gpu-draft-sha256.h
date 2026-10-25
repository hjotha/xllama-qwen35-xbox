#pragma once

#include <Windows.h>
#include <bcrypt.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

// Streaming checksum for the captured diagnostic input. CNG is available to
// UWP and avoids copying the 131 MB capture into application memory.
inline bool dflash_draft_sha256_file(const std::string& path, std::string* hex,
                                     std::string* error) {
    if (!hex || !error)
        return false;
    hex->clear();
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    auto close = [&]() {
        if (hash)
            BCryptDestroyHash(hash);
        if (algorithm)
            BCryptCloseAlgorithmProvider(algorithm, 0);
    };
    NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (status < 0) {
        *error = "SHA-256 algorithm unavailable";
        return false;
    }
    ULONG object_bytes = 0, digest_bytes = 0, returned = 0;
    status =
        BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_bytes),
                          sizeof(object_bytes), &returned, 0);
    if (status < 0 || returned != sizeof(object_bytes) || !object_bytes) {
        close();
        *error = "SHA-256 object size unavailable";
        return false;
    }
    status =
        BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&digest_bytes),
                          sizeof(digest_bytes), &returned, 0);
    if (status < 0 || returned != sizeof(digest_bytes) || digest_bytes != 32) {
        close();
        *error = "SHA-256 digest size invalid";
        return false;
    }
    std::vector<uint8_t> object(object_bytes);
    status = BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0);
    if (status < 0) {
        close();
        *error = "SHA-256 hash create failed";
        return false;
    }
    std::ifstream file(std::filesystem::path(path), std::ios::binary);
    if (!file) {
        close();
        *error = "capture hash open failed";
        return false;
    }
    std::vector<uint8_t> chunk(1u << 20);
    for (;;) {
        file.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
        const std::streamsize count = file.gcount();
        if (count > 0) {
            status = BCryptHashData(hash, chunk.data(), static_cast<ULONG>(count), 0);
            if (status < 0) {
                close();
                *error = "capture SHA-256 update failed";
                return false;
            }
        }
        if (count < static_cast<std::streamsize>(chunk.size())) {
            if (file.bad()) {
                close();
                *error = "capture SHA-256 read failed";
                return false;
            }
            break;
        }
    }
    std::array<uint8_t, 32> digest{};
    status = BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0);
    close();
    if (status < 0) {
        *error = "capture SHA-256 finish failed";
        return false;
    }
    constexpr char digits[] = "0123456789abcdef";
    hex->reserve(64);
    for (const uint8_t byte : digest) {
        hex->push_back(digits[byte >> 4]);
        hex->push_back(digits[byte & 15]);
    }
    error->clear();
    return true;
}
