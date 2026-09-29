#include "dedicated.h"
#include "remote-attn.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "llama.h"

#include <roapi.h>
#include <windows.h>

#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.Core.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Activation;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;

namespace {

std::atomic<int> server_state{0};
std::atomic<bool> closing{false};

void log_memory(std::string path) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    while (!closing.load()) {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        const auto process = winrt::Windows::System::MemoryManager::GetProcessMemoryReport();
        std::ofstream out(path, std::ios::app);
        out << "{\"timestampMs\":" << ms << ",\"serverState\":" << server_state.load()
            << ",\"appMemoryUsageBytes\":"
            << winrt::Windows::System::MemoryManager::AppMemoryUsage()
            << ",\"appMemoryUsageLimitBytes\":"
            << winrt::Windows::System::MemoryManager::AppMemoryUsageLimit()
            << ",\"privateWorkingSetBytes\":" << process.PrivateWorkingSetUsage() << "}\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void serve_rpc() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        server_state = -1;
        return;
    }
    server_state = 1;
    ggml_backend_rpc_start_server("0.0.0.0:50052", nullptr, 4, 1, &cpu);
    server_state = -2;
}

struct App : implements<App, IFrameworkViewSource, IFrameworkView> {
    IFrameworkView CreateView() {
        return *this;
    }

    void Initialize(CoreApplicationView const& view) {
        view.Activated([](CoreApplicationView const&, IActivatedEventArgs const&) {
            CoreWindow::GetForCurrentThread().Activate();
        });
    }

    void SetWindow(CoreWindow const& window) {
        window.Closed([](CoreWindow const&, CoreWindowEventArgs const&) { closing = true; });
    }

    void Load(hstring const&) {}

    void Run() {
        const auto local = winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path();
        const std::string path = winrt::to_string(local) + "\\rpc-memory.jsonl";
        llama_backend_init();
        std::thread(log_memory, path).detach();
        std::thread(serve_rpc).detach();
        std::thread(serve_dflash_dedicated, winrt::to_string(local)).detach();
        std::thread(serve_remote_attn, winrt::to_string(local)).detach();
        auto dispatcher = CoreWindow::GetForCurrentThread().Dispatcher();
        while (!closing.load()) {
            dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessOneAndAllPending);
        }
    }

    void Uninitialize() {
        closing = true;
    }
};

} // namespace

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::check_hresult(RoInitialize(RO_INIT_MULTITHREADED));
    winrt::init_apartment();
    CoreApplication::Run(winrt::make<App>());
    return 0;
}
