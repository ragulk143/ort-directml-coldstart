#include <windows.h>

#undef _In_
#undef _In_z_
#undef _In_opt_
#undef _In_opt_z_
#undef _Out_
#undef _Outptr_
#undef _Out_opt_
#undef _Inout_
#undef _Inout_opt_
#undef _Frees_ptr_opt_
#undef _Ret_maybenull_
#undef _Ret_notnull_
#undef _Check_return_
#undef _Success_
#undef _Out_writes_bytes_all_
#undef _In_reads_

#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Success_(x)
#define _Out_writes_bytes_all_(x)
#define _In_reads_(x)

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <onnxruntime_c_api.h>

using Clock = std::chrono::steady_clock;
const OrtApi* ort = nullptr;

void check_status(OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        std::cerr << "ORT Error: " << msg << "\n";
        ort->ReleaseStatus(status);
        std::exit(1);
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/squeezenet.onnx";
    int cpu_threads = 4;
    int gpu_device = 1;
    int total_frames = 300;
    int trial = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) cpu_threads = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) gpu_device = std::stoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc) total_frames = std::stoi(argv[++i]);
        else if (arg == "--trial" && i + 1 < argc) trial = std::stoi(argv[++i]);
    }

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env = nullptr;
    check_status(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HybridBench", &env));

    std::wstring w_model_path(model_path.begin(), model_path.end());

    // Prepare deterministic input tensor
    int64_t input_shape[] = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; ++i) {
        input_tensor_values[i] = static_cast<float>((i * 17) % 256) / 255.0f;
    }

    OrtMemoryInfo* memory_info = nullptr;
    check_status(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = nullptr;
    check_status(ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    ));

    auto app_start = Clock::now();

    // 1. Spawn asynchronous background thread to compile & warm up DirectML GPU session
    std::atomic<bool> gpu_ready{false};
    OrtSession* gpu_session = nullptr;
    OrtSessionOptions* gpu_opts = nullptr;
    double bg_gpu_init_ms = 0.0;

    std::thread gpu_worker([&]() {
        auto g0 = Clock::now();
        check_status(ort->CreateSessionOptions(&gpu_opts));
        check_status(ort->SetSessionGraphOptimizationLevel(gpu_opts, ORT_ENABLE_ALL));
        check_status(ort->DisableMemPattern(gpu_opts));
        check_status(ort->SetSessionExecutionMode(gpu_opts, ORT_SEQUENTIAL));
        std::string dev_str = std::to_string(gpu_device);
        const char* keys[] = { "device_id" };
        const char* vals[] = { dev_str.c_str() };
        check_status(ort->SessionOptionsAppendExecutionProvider(gpu_opts, "DML", keys, vals, 1));
        check_status(ort->CreateSession(env, w_model_path.c_str(), gpu_opts, &gpu_session));

        // Perform 1 dummy warm-up inference on GPU so frame handoff has zero spike
        OrtAllocator* alloc = nullptr;
        check_status(ort->GetAllocatorWithDefaultOptions(&alloc));
        char* in_n = nullptr;
        char* out_n = nullptr;
        check_status(ort->SessionGetInputName(gpu_session, 0, alloc, &in_n));
        check_status(ort->SessionGetOutputName(gpu_session, 0, alloc, &out_n));
        const char* ins[] = { in_n };
        const char* outs[] = { out_n };

        OrtValue* dummy_out = nullptr;
        check_status(ort->Run(gpu_session, nullptr, ins, (const OrtValue* const*)&input_tensor, 1, outs, 1, &dummy_out));
        ort->ReleaseValue(dummy_out);
        check_status(ort->AllocatorFree(alloc, in_n));
        check_status(ort->AllocatorFree(alloc, out_n));

        auto g1 = Clock::now();
        bg_gpu_init_ms = elapsed_ms(g0, g1);
        gpu_ready.store(true, std::memory_order_release);
    });

    // 2. Immediately initialize fast CPU session on main thread
    OrtSessionOptions* cpu_opts = nullptr;
    check_status(ort->CreateSessionOptions(&cpu_opts));
    check_status(ort->SetSessionGraphOptimizationLevel(cpu_opts, ORT_ENABLE_ALL));
    check_status(ort->SetIntraOpNumThreads(cpu_opts, cpu_threads));

    OrtSession* cpu_session = nullptr;
    check_status(ort->CreateSession(env, w_model_path.c_str(), cpu_opts, &cpu_session));

    OrtAllocator* allocator = nullptr;
    check_status(ort->GetAllocatorWithDefaultOptions(&allocator));
    char* input_name = nullptr;
    char* output_name = nullptr;
    check_status(ort->SessionGetInputName(cpu_session, 0, allocator, &input_name));
    check_status(ort->SessionGetOutputName(cpu_session, 0, allocator, &output_name));
    const char* input_names[] = { input_name };
    const char* output_names[] = { output_name };

    // 3. Process stream of frames, atomically switching to GPU when ready
    double ttfi_ms = 0.0;
    int switch_frame = -1;
    std::vector<double> cpu_frame_times;
    std::vector<double> gpu_frame_times;

    for (int f = 0; f < total_frames; ++f) {
        bool use_gpu = gpu_ready.load(std::memory_order_acquire);
        if (use_gpu && switch_frame == -1) {
            switch_frame = f;
        }
        OrtSession* active_session = use_gpu ? gpu_session : cpu_session;

        OrtValue* out_val = nullptr;
        auto fs = Clock::now();
        check_status(ort->Run(active_session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &out_val));
        auto fe = Clock::now();

        if (f == 0) {
            ttfi_ms = elapsed_ms(app_start, fe);
        }
        double frame_ms = elapsed_ms(fs, fe);
        if (use_gpu) gpu_frame_times.push_back(frame_ms);
        else cpu_frame_times.push_back(frame_ms);

        ort->ReleaseValue(out_val);
    }

    auto app_end = Clock::now();
    double total_wall_ms = elapsed_ms(app_start, app_end);

    gpu_worker.join();

    double cpu_med = percentile(cpu_frame_times, 50.0);
    double gpu_med = percentile(gpu_frame_times, 50.0);
    double overall_p95 = percentile(
        [&]() {
            std::vector<double> all = cpu_frame_times;
            all.insert(all.end(), gpu_frame_times.begin(), gpu_frame_times.end());
            return all;
        }(), 95.0
    );

    std::cout << std::fixed << std::setprecision(4)
              << model_path << ","
              << gpu_device << ","
              << total_frames << ","
              << trial << ","
              << ttfi_ms << ","
              << switch_frame << ","
              << bg_gpu_init_ms << ","
              << cpu_med << ","
              << gpu_med << ","
              << overall_p95 << ","
              << total_wall_ms << "\n";

    ort->ReleaseValue(input_tensor);
    ort->ReleaseMemoryInfo(memory_info);
    check_status(ort->AllocatorFree(allocator, input_name));
    check_status(ort->AllocatorFree(allocator, output_name));
    ort->ReleaseSession(cpu_session);
    ort->ReleaseSessionOptions(cpu_opts);
    ort->ReleaseSession(gpu_session);
    ort->ReleaseSessionOptions(gpu_opts);
    ort->ReleaseEnv(env);
    return 0;
}
