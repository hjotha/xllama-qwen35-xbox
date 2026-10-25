from pathlib import Path

path = Path(r"C:\Users\hjotha\worktrees\xllama-dflash-xbox\uwp\dflash-rpc\dflash-gpu-engine.cpp")
text = path.read_text(encoding="utf-8")

target = """        Buffer upload;
        if (!create_buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &upload, error) ||
            !create_buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &output->buffer, error))
            return false;
        void * mapped = nullptr;
        HRESULT hr = upload.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) return fail(error, "GGUF upload Map", hr);
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        file.read(static_cast<char *>(mapped), static_cast<std::streamsize>(bytes));
        const bool read_ok = static_cast<bool>(file);
        if (aligned4(bytes) > bytes)
            std::memset(static_cast<uint8_t *>(mapped) + bytes, 0,
                        static_cast<size_t>(aligned4(bytes) - bytes));
        upload.resource->Unmap(0, nullptr);
        if (!read_ok) return fail(error, ("GGUF tensor read failed " + name).c_str());
        cmd_->CopyBufferRegion(output->buffer.resource.Get(), 0,
                               upload.resource.Get(), 0, aligned4(bytes));"""

replacement = """        const uint64_t n_blocks = (type != GGML_TYPE_F32) ? (uint64_t(cols / 256) * rows) : 0u;
        const uint64_t gpu_payload_bytes = (type == GGML_TYPE_Q2_K) ? (n_blocks * 288u) : bytes;
        Buffer upload;
        if (!create_buffer(gpu_payload_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_GENERIC_READ, &upload, error) ||
            !create_buffer(gpu_payload_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, &output->buffer, error))
            return false;
        void * mapped = nullptr;
        HRESULT hr = upload.resource->Map(0, nullptr, &mapped);
        if (FAILED(hr)) return fail(error, "GGUF upload Map", hr);
        bool read_ok = false;
        if (type == GGML_TYPE_Q2_K) {
            std::vector<uint8_t> raw_q2(static_cast<size_t>(bytes));
            file.clear();
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(reinterpret_cast<char *>(raw_q2.data()), static_cast<std::streamsize>(bytes));
            read_ok = static_cast<bool>(file);
            if (read_ok) {
                auto * dst = reinterpret_cast<uint8_t *>(mapped);
                for (uint64_t b = 0; b < n_blocks; ++b) {
                    const uint8_t * src_block = raw_q2.data() + b * 84u;
                    uint8_t * dst_block = dst + b * 288u;
                    const uint8_t * scales = src_block;
                    const uint8_t * qs = src_block + 16;
                    uint16_t d_u16, dmin_u16;
                    std::memcpy(&d_u16, src_block + 80, 2);
                    std::memcpy(&dmin_u16, src_block + 82, 2);
                    const float d_f32 = DirectX::PackedVector::XMConvertHalfToFloat(d_u16);
                    const float dmin_f32 = DirectX::PackedVector::XMConvertHalfToFloat(dmin_u16);

                    for (uint32_t element = 0; element < 256; ++element) {
                        const uint32_t half = element >> 7u;
                        const uint32_t quarter = (element >> 5u) & 3u;
                        const uint32_t quant = (qs[half * 32u + (element & 31u)] >> (quarter * 2u)) & 3u;
                        const uint32_t scale = scales[element >> 4u];
                        dst_block[element] = static_cast<uint8_t>((scale & 15u) * quant);
                    }
                    for (uint32_t i = 0; i < 16; ++i) {
                        dst_block[256 + i] = static_cast<uint8_t>(scales[i] >> 4u);
                    }
                    std::memcpy(dst_block + 272, &d_f32, 4);
                    std::memcpy(dst_block + 276, &dmin_f32, 4);
                    std::memset(dst_block + 280, 0, 8);
                }
            }
        } else {
            file.clear();
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(static_cast<char *>(mapped), static_cast<std::streamsize>(bytes));
            read_ok = static_cast<bool>(file);
        }
        if (aligned4(gpu_payload_bytes) > gpu_payload_bytes)
            std::memset(static_cast<uint8_t *>(mapped) + gpu_payload_bytes, 0,
                        static_cast<size_t>(aligned4(gpu_payload_bytes) - gpu_payload_bytes));
        upload.resource->Unmap(0, nullptr);
        if (!read_ok) return fail(error, ("GGUF tensor read failed " + name).c_str());
        cmd_->CopyBufferRegion(output->buffer.resource.Get(), 0,
                               upload.resource.Get(), 0, aligned4(gpu_payload_bytes));"""

norm_target = target.replace("\r\n", "\n")
norm_replacement = replacement.replace("\r\n", "\n")
norm_text = text.replace("\r\n", "\n")

if norm_target in norm_text:
    new_text = norm_text.replace(norm_target, norm_replacement, 1)
    if "\r\n" in text:
        new_text = new_text.replace("\n", "\r\n")
    path.write_text(new_text, encoding="utf-8")
    print("SUCCESS")
else:
    print("TARGET NOT FOUND")
