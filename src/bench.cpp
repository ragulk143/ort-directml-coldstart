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
    std::string ep = "cpu";
    int threads = 4;
    int device_id = 0;
    int runs = 50;
    int trial = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--ep" && i + 1 < argc) ep = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_id = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
        else if (arg == "--trial" && i + 1 < argc) trial = std::stoi(argv[++i]);
    }

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env = nullptr;
    check_status(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ColdStartBench", &env));

    OrtSessionOptions* session_options = nullptr;
    check_status(ort->CreateSessionOptions(&session_options));
    check_status(ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

    if (ep == "cpu") {
        check_status(ort->SetIntraOpNumThreads(session_options, threads));
    } else if (ep == "dml") {
        check_status(ort->DisableMemPattern(session_options));
        check_status(ort->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL));
        std::string dev_str = std::to_string(device_id);
        const char* keys[] = { "device_id" };
        const char* vals[] = { dev_str.c_str() };
        check_status(ort->SessionOptionsAppendExecutionProvider(session_options, "DML", keys, vals, 1));
    }

    std::wstring w_model_path(model_path.begin(), model_path.end());

    OrtSession* session = nullptr;
    auto t0 = Clock::now();
    check_status(ort->CreateSession(env, w_model_path.c_str(), session_options, &session));
    auto t1 = Clock::now();
    double session_create_ms = elapsed_ms(t0, t1);

    OrtAllocator* allocator = nullptr;
    check_status(ort->GetAllocatorWithDefaultOptions(&allocator));

    char* input_name = nullptr;
    char* output_name = nullptr;
    check_status(ort->SessionGetInputName(session, 0, allocator, &input_name));
    check_status(ort->SessionGetOutputName(session, 0, allocator, &output_name));
    const char* input_names[] = { input_name };
    const char* output_names[] = { output_name };

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

    // Cold first inference
    OrtValue* output_tensor = nullptr;
    auto t2 = Clock::now();
    check_status(ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
    auto t3 = Clock::now();
    double first_inf_ms = elapsed_ms(t2, t3);

    // Checksum for numerical verification
    float* floatarr = nullptr;
    check_status(ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    OrtTensorTypeAndShapeInfo* out_info = nullptr;
    check_status(ort->GetTensorTypeAndShape(output_tensor, &out_info));
    size_t out_count = 0;
    check_status(ort->GetTensorShapeElementCount(out_info, &out_count));
    double checksum = 0.0;
    for (size_t i = 0; i < out_count; ++i) checksum += floatarr[i];
    ort->ReleaseTensorTypeAndShapeInfo(out_info);
    ort->ReleaseValue(output_tensor);

    // Steady-state warm runs
    std::vector<double> warm_times;
    warm_times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        OrtValue* out_warm = nullptr;
        auto ts = Clock::now();
        check_status(ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &out_warm));
        auto te = Clock::now();
        warm_times.push_back(elapsed_ms(ts, te));
        ort->ReleaseValue(out_warm);
    }

    double steady_median_ms = percentile(warm_times, 50.0);
    double steady_p95_ms = percentile(warm_times, 95.0);
    double cold_gap_ms = (session_create_ms + first_inf_ms) - steady_median_ms;

    std::cout << std::fixed << std::setprecision(4)
              << model_path << ","
              << ep << ","
              << (ep == "cpu" ? threads : device_id) << ","
              << trial << ","
              << session_create_ms << ","
              << first_inf_ms << ","
              << steady_median_ms << ","
              << steady_p95_ms << ","
              << cold_gap_ms << ","
              << checksum << "\n";

    ort->ReleaseValue(input_tensor);
    ort->ReleaseMemoryInfo(memory_info);
    check_status(ort->AllocatorFree(allocator, input_name));
    check_status(ort->AllocatorFree(allocator, output_name));
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(session_options);
    ort->ReleaseEnv(env);
    return 0;
}
