#include "dflash-remote.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2)
        return 2;
    try {
        dflash_remote::client rpc(argv[1], argc > 2 ? (uint16_t)std::atoi(argv[2]) : 50053);
        std::vector<uint8_t> output;
        uint64_t id = 0;
        rpc.call(dflash_remote::hello, ++id, 0, nullptr, 0, output,
                 sizeof(dflash_remote::hello_data));
#ifdef TEST_FUSED_PROBE
        bool supported = true;
        try {
            rpc.call(dflash_remote::sync_trim, ++id, 0, nullptr, 0, output, 0);
        } catch (const dflash_remote::server_error& e) {
            if (e.status != -1)
                throw;
            supported = false;
        }
        std::cout << "{\"fused_supported\":" << (supported ? "true" : "false") << "}\n";
#endif
        rpc.call(dflash_remote::reset, ++id, 0, nullptr, 0, output, 0);
        for (int i = 0; i < 100; ++i) {
            const dflash_remote::trim_data cut{7};
            const auto t = rpc.call(dflash_remote::trim, ++id, 0, &cut, sizeof(cut), output, 0);
            std::cout << "{\"i\":" << i << ",\"roundtrip_us\":" << t.roundtrip_us
                      << ",\"xbox_receive_us\":" << t.server.xbox_receive_us
                      << ",\"xbox_prepare_us\":" << t.server.xbox_prepare_us << "}\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
