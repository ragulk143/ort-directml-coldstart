
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

#include <onnxruntime_c_api.h>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <cctype>

using Clock = std::chrono::steady_clock;

const OrtApi* ort = nullptr;

void check_status(OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = ort->GetErrorMessage(status);
        std::cerr << "ONNX Runtime error: " << msg << "\n";
        ort->ReleaseStatus(status);
        std::exit(EXIT_FAILURE);
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::sort(values.begin(), values.end());

    double index = (p / 100.0) * (values.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(index));
    size_t hi = static_cast<size_t>(std::ceil(index));

    if (lo == hi) return values[lo];

    return values[lo] +
        (values[hi] - values[lo]) * (index - lo);
}

std::string lower(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );
    return value;
}

void get_default_estimates(
    const std::string& model,
    int device,
    double& cpu_ms,
    double& gpu_ms
) {
    const std::string name = lower(model);

    // Historical four-thread CPU and DirectML medians.
    // Device 0 = Intel UHD
    // Device 1 = NVIDIA RTX 3050

    if (name.find("squeezenet") != std::string::npos) {
        cpu_ms = 2.37;
        gpu_ms = (device == 0) ? 1.03 : 3.89;
    }
    else if (name.find("mobilenetv2") != std::string::npos) {
        cpu_ms = 2.72;
        gpu_ms = (device == 0) ? 4.07 : 12.47;
    }
    else {
        std::cerr
            << "Unknown model. Supply --cpu-ms and --gpu-ms.\n";
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/squeezenet.onnx";

    int cpu_threads = 4;
    int gpu_device = 1;
    int total_frames = 800;
    int trial = 1;

    double cpu_estimate_ms =
        std::numeric_limits<double>::quiet_NaN();

    double gpu_estimate_ms =
        std::numeric_limits<double>::quiet_NaN();

    double handoff_overhead_ms = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];

        else if (arg == "--threads" && i + 1 < argc)
            cpu_threads = std::stoi(argv[++i]);

        else if (arg == "--device" && i + 1 < argc)
            gpu_device = std::stoi(argv[++i]);

        else if (arg == "--frames" && i + 1 < argc)
            total_frames = std::stoi(argv[++i]);

        else if (arg == "--trial" && i + 1 < argc)
            trial = std::stoi(argv[++i]);

        else if (arg == "--cpu-ms" && i + 1 < argc)
            cpu_estimate_ms = std::stod(argv[++i]);

        else if (arg == "--gpu-ms" && i + 1 < argc)
            gpu_estimate_ms = std::stod(argv[++i]);

        else if (arg == "--handoff-ms" && i + 1 < argc)
            handoff_overhead_ms = std::stod(argv[++i]);

        else {
            std::cerr << "Unknown or incomplete argument: "
                      << arg << "\n";
            return EXIT_FAILURE;
        }
    }

    if (total_frames <= 0 || cpu_threads <= 0 ||
        gpu_device < 0 || handoff_overhead_ms < 0.0) {
        std::cerr << "Invalid benchmark configuration.\n";
        return EXIT_FAILURE;
    }

    if (std::isnan(cpu_estimate_ms) ||
        std::isnan(gpu_estimate_ms)) {
        get_default_estimates(
            model_path,
            gpu_device,
            cpu_estimate_ms,
            gpu_estimate_ms
        );
    }

    if (cpu_estimate_ms <= 0.0 || gpu_estimate_ms <= 0.0) {
        std::cerr << "Latency estimates must be positive.\n";
        return EXIT_FAILURE;
    }

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    if (ort == nullptr) {
        std::cerr << "Could not obtain ONNX Runtime API.\n";
        return EXIT_FAILURE;
    }

    OrtEnv* env = nullptr;
    check_status(
        ort->CreateEnv(
            ORT_LOGGING_LEVEL_WARNING,
            "CrossoverHybridBench",
            &env
        )
    );

    std::wstring w_model_path(
        model_path.begin(), model_path.end()
    );

    // Deterministic input tensor.
    int64_t input_shape[] = {1, 3, 224, 224};
    const size_t input_count = 1 * 3 * 224 * 224;

    std::vector<float> input_values(input_count);

    for (size_t i = 0; i < input_count; ++i) {
        input_values[i] =
            static_cast<float>((i * 17) % 256) / 255.0f;
    }

    OrtMemoryInfo* memory_info = nullptr;

    check_status(
        ort->CreateCpuMemoryInfo(
            OrtArenaAllocator,
            OrtMemTypeDefault,
            &memory_info
        )
    );

    OrtValue* input_tensor = nullptr;

    check_status(
        ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_values.data(),
            input_count * sizeof(float),
            input_shape,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        )
    );

    const auto app_start = Clock::now();

    // CPU session is always initialized first.
    OrtSessionOptions* cpu_options = nullptr;

    check_status(ort->CreateSessionOptions(&cpu_options));
    check_status(
        ort->SetSessionGraphOptimizationLevel(
            cpu_options, ORT_ENABLE_ALL
        )
    );
    check_status(
        ort->SetIntraOpNumThreads(cpu_options, cpu_threads)
    );

    OrtSession* cpu_session = nullptr;

    check_status(
        ort->CreateSession(
            env,
            w_model_path.c_str(),
            cpu_options,
            &cpu_session
        )
    );

    OrtAllocator* allocator = nullptr;

    check_status(
        ort->GetAllocatorWithDefaultOptions(&allocator)
    );

    char* cpu_input_name = nullptr;
    char* cpu_output_name = nullptr;

    check_status(
        ort->SessionGetInputName(
            cpu_session, 0, allocator, &cpu_input_name
        )
    );

    check_status(
        ort->SessionGetOutputName(
            cpu_session, 0, allocator, &cpu_output_name
        )
    );

    const char* cpu_inputs[] = {cpu_input_name};
    const char* cpu_outputs[] = {cpu_output_name};

    // The GPU is a candidate only if its measured steady-state
    // latency is lower than the CPU estimate.
    const bool gpu_eligible =
        gpu_estimate_ms < cpu_estimate_ms;

    std::atomic<bool> gpu_ready{false};

    OrtSession* gpu_session = nullptr;
    OrtSessionOptions* gpu_options = nullptr;

    double bg_gpu_init_ms =
        std::numeric_limits<double>::quiet_NaN();

    std::thread gpu_worker;

    if (gpu_eligible) {
        gpu_worker = std::thread([&]() {
            const auto gpu_start = Clock::now();

            check_status(
                ort->CreateSessionOptions(&gpu_options)
            );

            check_status(
                ort->SetSessionGraphOptimizationLevel(
                    gpu_options, ORT_ENABLE_ALL
                )
            );

            check_status(ort->DisableMemPattern(gpu_options));

            check_status(
                ort->SetSessionExecutionMode(
                    gpu_options, ORT_SEQUENTIAL
                )
            );

            std::string device_string =
                std::to_string(gpu_device);

            const char* keys[] = {"device_id"};
            const char* values[] = {device_string.c_str()};

            check_status(
                ort->SessionOptionsAppendExecutionProvider(
                    gpu_options,
                    "DML",
                    keys,
                    values,
                    1
                )
            );

            check_status(
                ort->CreateSession(
                    env,
                    w_model_path.c_str(),
                    gpu_options,
                    &gpu_session
                )
            );

            // Retrieve GPU session names independently.
            char* gpu_input_name = nullptr;
            char* gpu_output_name = nullptr;

            check_status(
                ort->SessionGetInputName(
                    gpu_session, 0, allocator, &gpu_input_name
                )
            );

            check_status(
                ort->SessionGetOutputName(
                    gpu_session, 0, allocator, &gpu_output_name
                )
            );

            const char* gpu_inputs[] = {gpu_input_name};
            const char* gpu_outputs[] = {gpu_output_name};

            // One warm-up inference before publishing readiness.
            OrtValue* warm_output = nullptr;

            check_status(
                ort->Run(
                    gpu_session,
                    nullptr,
                    gpu_inputs,
                    (const OrtValue* const*)&input_tensor,
                    1,
                    gpu_outputs,
                    1,
                    &warm_output
                )
            );

            ort->ReleaseValue(warm_output);

            check_status(
                ort->AllocatorFree(allocator, gpu_input_name)
            );

            check_status(
                ort->AllocatorFree(allocator, gpu_output_name)
            );

            bg_gpu_init_ms =
                elapsed_ms(gpu_start, Clock::now());

            // Publish only after session creation and warm-up.
            gpu_ready.store(true, std::memory_order_release);
        });
    }

    // Workload execution.
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;

    cpu_times.reserve(total_frames);
    gpu_times.reserve(total_frames);

    double ttfi_ms = 0.0;
    double predicted_cpu_remaining_ms = 0.0;
    double predicted_gpu_remaining_ms = 0.0;

    int switch_frame = -1;

    bool use_gpu = false;
    bool decision_made = false;

    for (int frame = 0; frame < total_frames; ++frame) {
        if (!decision_made && gpu_eligible &&
            gpu_ready.load(std::memory_order_acquire)) {

            const int remaining = total_frames - frame;

            predicted_cpu_remaining_ms =
                remaining * cpu_estimate_ms;

            predicted_gpu_remaining_ms =
                remaining * gpu_estimate_ms +
                handoff_overhead_ms;

            // Strictly select GPU only when predicted faster.
            use_gpu =
                predicted_gpu_remaining_ms <
                predicted_cpu_remaining_ms;

            decision_made = true;

            if (use_gpu) {
                switch_frame = frame;
            }
        }

        OrtSession* active_session =
            use_gpu ? gpu_session : cpu_session;

        const char* const* active_inputs =
            use_gpu ? nullptr : cpu_inputs;

        const char* const* active_outputs =
            use_gpu ? nullptr : cpu_outputs;

        // GPU names are retrieved in the worker. Store their
        // strings here after readiness to keep names alive.
        static std::string gpu_input_storage;
        static std::string gpu_output_storage;

        if (use_gpu && frame == switch_frame) {
            char* name = nullptr;

            check_status(
                ort->SessionGetInputName(
                    gpu_session, 0, allocator, &name
                )
            );

            gpu_input_storage = name;

            check_status(ort->AllocatorFree(allocator, name));

            check_status(
                ort->SessionGetOutputName(
                    gpu_session, 0, allocator, &name
                )
            );

            gpu_output_storage = name;

            check_status(ort->AllocatorFree(allocator, name));
        }

        const char* gpu_inputs[] = {
            gpu_input_storage.c_str()
        };

        const char* gpu_outputs[] = {
            gpu_output_storage.c_str()
        };

        if (use_gpu) {
            active_inputs = gpu_inputs;
            active_outputs = gpu_outputs;
        }

        OrtValue* output = nullptr;

        const auto frame_start = Clock::now();

        check_status(
            ort->Run(
                active_session,
                nullptr,
                active_inputs,
                (const OrtValue* const*)&input_tensor,
                1,
                active_outputs,
                1,
                &output
            )
        );

        const auto frame_end = Clock::now();

        const double frame_ms =
            elapsed_ms(frame_start, frame_end);

        if (frame == 0) {
            ttfi_ms = elapsed_ms(app_start, frame_end);
        }

        if (use_gpu) {
            gpu_times.push_back(frame_ms);
        } else {
            cpu_times.push_back(frame_ms);
        }

        ort->ReleaseValue(output);
    }

    // Workload completion excludes any remaining GPU worker
    // initialization after the last CPU frame.
    const auto workload_end = Clock::now();

    const double workload_wall_ms =
        elapsed_ms(app_start, workload_end);

    if (gpu_worker.joinable()) {
        gpu_worker.join();
    }

    // Process completion includes the worker's final completion.
    const auto process_end = Clock::now();

    const double process_wall_ms =
        elapsed_ms(app_start, process_end);

    const double cpu_median =
        percentile(cpu_times, 50.0);

    const double gpu_median =
        percentile(gpu_times, 50.0);

    std::vector<double> all_times = cpu_times;

    all_times.insert(
        all_times.end(),
        gpu_times.begin(),
        gpu_times.end()
    );

    const double overall_p95 =
        percentile(all_times, 95.0);

    std::string decision;

if (!gpu_eligible) {
    decision = "skipped_unprofitable";
}
else if (switch_frame >= 0) {
    decision = "gpu_selected";
}
else if (decision_made) {
    decision = "cpu_retained";
}
else {
    decision = "gpu_not_ready_before_end";
}

    // CSV output. Keep diagnostics on stderr so stdout remains
    // machine-readable.
    std::cout
        << "model,gpu_device,total_frames,trial,decision,"
        << "ttfi_ms,switch_frame,bg_gpu_init_ms,"
        << "cpu_phase_med_ms,gpu_phase_med_ms,overall_p95_ms,"
        << "workload_wall_ms,process_wall_ms,"
        << "cpu_estimate_ms,gpu_estimate_ms,"
        << "pred_cpu_remaining_ms,pred_gpu_remaining_ms,"
        << "handoff_overhead_ms\n";

    std::cout << std::fixed << std::setprecision(4)
        << model_path << ","
        << gpu_device << ","
        << total_frames << ","
        << trial << ","
        << decision << ","
        << ttfi_ms << ","
        << switch_frame << ","
        << bg_gpu_init_ms << ","
        << cpu_median << ","
        << gpu_median << ","
        << overall_p95 << ","
        << workload_wall_ms << ","
        << process_wall_ms << ","
        << cpu_estimate_ms << ","
        << gpu_estimate_ms << ","
        << predicted_cpu_remaining_ms << ","
        << predicted_gpu_remaining_ms << ","
        << handoff_overhead_ms << "\n";

    // Cleanup after worker termination.
    ort->ReleaseValue(input_tensor);
    ort->ReleaseMemoryInfo(memory_info);

    check_status(
        ort->AllocatorFree(allocator, cpu_input_name)
    );

    check_status(
        ort->AllocatorFree(allocator, cpu_output_name)
    );

    if (gpu_session != nullptr) {
        ort->ReleaseSession(gpu_session);
    }

    if (gpu_options != nullptr) {
        ort->ReleaseSessionOptions(gpu_options);
    }

    ort->ReleaseSession(cpu_session);
    ort->ReleaseSessionOptions(cpu_options);
    ort->ReleaseEnv(env);

    return EXIT_SUCCESS;
}
