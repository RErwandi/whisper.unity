#include <windows.h>

#include <array>
#include <exception>
#include <mutex>
#include <string>

#include <whisper.h>

namespace
{
    enum class BackendKind
    {
        None,
        Cpu,
        Vulkan,
    };

    using FnWhisperInitFromFileWithParams = whisper_context * (*)(const char *, whisper_context_params);
    using FnWhisperInitFromBufferWithParams = whisper_context * (*)(void *, size_t, whisper_context_params);
    using FnWhisperVersion = const char * (*)();
    using FnWhisperLangMaxId = int (*)();
    using FnWhisperLangId = int (*)(const char *);
    using FnWhisperLangStr = const char * (*)(int);
    using FnWhisperTokenEot = whisper_token (*)(whisper_context *);
    using FnWhisperPrintSystemInfo = const char * (*)();
    using FnWhisperFullDefaultParams = whisper_full_params (*)(whisper_sampling_strategy);
    using FnWhisperContextDefaultParams = whisper_context_params (*)();
    using FnWhisperFull = int (*)(whisper_context *, whisper_full_params, const float *, int);
    using FnWhisperFullNSegments = int (*)(whisper_context *);
    using FnWhisperFullLangId = int (*)(whisper_context *);
    using FnWhisperIsMultilingual = int (*)(whisper_context *);
    using FnWhisperFullGetSegmentT0 = int64_t (*)(whisper_context *, int);
    using FnWhisperFullGetSegmentT1 = int64_t (*)(whisper_context *, int);
    using FnWhisperFullGetSegmentText = const char * (*)(whisper_context *, int);
    using FnWhisperFullNTokens = int (*)(whisper_context *, int);
    using FnWhisperFullGetTokenText = const char * (*)(whisper_context *, int, int);
    using FnWhisperFullGetTokenData = whisper_token_data (*)(whisper_context *, int, int);
    using FnWhisperFree = void (*)(whisper_context *);

    struct BackendApi
    {
        BackendKind Kind = BackendKind::None;
        std::array<HMODULE, 5> Modules{};
        int ModuleCount = 0;
        HMODULE WhisperModule = nullptr;

        FnWhisperInitFromFileWithParams InitFromFileWithParams = nullptr;
        FnWhisperInitFromBufferWithParams InitFromBufferWithParams = nullptr;
        FnWhisperVersion Version = nullptr;
        FnWhisperLangMaxId LangMaxId = nullptr;
        FnWhisperLangId LangId = nullptr;
        FnWhisperLangStr LangStr = nullptr;
        FnWhisperTokenEot TokenEot = nullptr;
        FnWhisperPrintSystemInfo PrintSystemInfo = nullptr;
        FnWhisperFullDefaultParams FullDefaultParams = nullptr;
        FnWhisperContextDefaultParams ContextDefaultParams = nullptr;
        FnWhisperFull Full = nullptr;
        FnWhisperFullNSegments FullNSegments = nullptr;
        FnWhisperFullLangId FullLangId = nullptr;
        FnWhisperIsMultilingual IsMultilingual = nullptr;
        FnWhisperFullGetSegmentT0 FullGetSegmentT0 = nullptr;
        FnWhisperFullGetSegmentT1 FullGetSegmentT1 = nullptr;
        FnWhisperFullGetSegmentText FullGetSegmentText = nullptr;
        FnWhisperFullNTokens FullNTokens = nullptr;
        FnWhisperFullGetTokenText FullGetTokenText = nullptr;
        FnWhisperFullGetTokenData FullGetTokenData = nullptr;
        FnWhisperFree Free = nullptr;
    };

    HMODULE g_selfModule = nullptr;
    BackendApi g_activeBackend{};
    std::mutex g_backendMutex;
    std::string g_lastBackendError;
    std::string g_lastBackendName;
    std::string g_stringScratch;

    std::wstring GetModuleDirectory()
    {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(g_selfModule, path, MAX_PATH);
        std::wstring directory(path);
        const auto slashIndex = directory.find_last_of(L"\\/");
        if (slashIndex == std::wstring::npos)
            return L".";

        directory.resize(slashIndex);
        return directory;
    }

    std::wstring GetBackendDirectory(BackendKind kind)
    {
        auto directory = GetModuleDirectory();
        directory += L"\\backends\\";
        directory += kind == BackendKind::Vulkan ? L"vulkan" : L"cpu";
        return directory;
    }

    std::wstring JoinPath(const std::wstring & left, const std::wstring & right)
    {
        if (left.empty())
            return right;

        if (left.back() == L'\\' || left.back() == L'/')
            return left + right;

        return left + L"\\" + right;
    }

    std::string WideToUtf8(const std::wstring & value)
    {
        if (value.empty())
            return {};

        const auto length = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if (length <= 1)
            return {};

        std::string result(static_cast<size_t>(length), '\0');
        WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, result.data(), length, nullptr, nullptr);
        result.resize(static_cast<size_t>(length - 1));
        return result;
    }

    std::string GetLastWin32ErrorString()
    {
        const auto errorCode = GetLastError();
        if (errorCode == 0)
            return {};

        wchar_t * buffer = nullptr;
        const auto size = FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            errorCode,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<LPWSTR>(&buffer),
            0,
            nullptr);

        std::wstring message;
        if (size > 0 && buffer != nullptr)
        {
            message.assign(buffer, size);
            LocalFree(buffer);
        }

        return WideToUtf8(message);
    }

    void SetLastBackendError(std::string message)
    {
        g_lastBackendError = std::move(message);
    }

    void ClearLastBackendError()
    {
        g_lastBackendError.clear();
    }

    const char * CopyScratch(const char * value)
    {
        if (value == nullptr)
        {
            g_stringScratch.clear();
            return nullptr;
        }

        g_stringScratch = value;
        return g_stringScratch.c_str();
    }

    template <typename TFn>
    bool ResolveFunction(HMODULE module, const char * name, TFn & target)
    {
        target = reinterpret_cast<TFn>(GetProcAddress(module, name));
        if (target != nullptr)
            return true;

        SetLastBackendError("Missing native export: " + std::string(name));
        return false;
    }

    void ResetBackend(BackendApi & backend)
    {
        backend = BackendApi{};
    }

    void UnloadBackend(BackendApi & backend)
    {
        for (auto index = backend.ModuleCount - 1; index >= 0; index--)
        {
            if (backend.Modules[index] != nullptr)
            {
                FreeLibrary(backend.Modules[index]);
                backend.Modules[index] = nullptr;
            }
        }

        ResetBackend(backend);
    }

    bool LoadModuleFromFolder(BackendApi & backend, const std::wstring & folder, const wchar_t * fileName, bool required)
    {
        const auto fullPath = JoinPath(folder, fileName);
        const auto module = LoadLibraryExW(
            fullPath.c_str(),
            nullptr,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
        if (module == nullptr)
        {
            if (required)
            {
                SetLastBackendError(
                    "Failed to load '" + WideToUtf8(fullPath) + "': " + GetLastWin32ErrorString());
                return false;
            }

            return true;
        }

        backend.Modules[backend.ModuleCount++] = module;
        return true;
    }

    bool ResolveBackendFunctions(BackendApi & backend)
    {
        return
            ResolveFunction(backend.WhisperModule, "whisper_init_from_file_with_params", backend.InitFromFileWithParams) &&
            ResolveFunction(backend.WhisperModule, "whisper_init_from_buffer_with_params", backend.InitFromBufferWithParams) &&
            ResolveFunction(backend.WhisperModule, "whisper_version", backend.Version) &&
            ResolveFunction(backend.WhisperModule, "whisper_lang_max_id", backend.LangMaxId) &&
            ResolveFunction(backend.WhisperModule, "whisper_lang_id", backend.LangId) &&
            ResolveFunction(backend.WhisperModule, "whisper_lang_str", backend.LangStr) &&
            ResolveFunction(backend.WhisperModule, "whisper_token_eot", backend.TokenEot) &&
            ResolveFunction(backend.WhisperModule, "whisper_print_system_info", backend.PrintSystemInfo) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_default_params", backend.FullDefaultParams) &&
            ResolveFunction(backend.WhisperModule, "whisper_context_default_params", backend.ContextDefaultParams) &&
            ResolveFunction(backend.WhisperModule, "whisper_full", backend.Full) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_n_segments", backend.FullNSegments) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_lang_id", backend.FullLangId) &&
            ResolveFunction(backend.WhisperModule, "whisper_is_multilingual", backend.IsMultilingual) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_get_segment_t0", backend.FullGetSegmentT0) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_get_segment_t1", backend.FullGetSegmentT1) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_get_segment_text", backend.FullGetSegmentText) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_n_tokens", backend.FullNTokens) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_get_token_text", backend.FullGetTokenText) &&
            ResolveFunction(backend.WhisperModule, "whisper_full_get_token_data", backend.FullGetTokenData) &&
            ResolveFunction(backend.WhisperModule, "whisper_free", backend.Free);
    }

    bool LoadBackend(BackendApi & backend, BackendKind kind)
    {
        ResetBackend(backend);
        backend.Kind = kind;

        const auto folder = GetBackendDirectory(kind);
        if (!LoadModuleFromFolder(backend, folder, L"ggml-base.dll", true) ||
            !LoadModuleFromFolder(backend, folder, L"ggml.dll", true) ||
            !LoadModuleFromFolder(backend, folder, L"ggml-cpu.dll", true))
        {
            UnloadBackend(backend);
            return false;
        }

        if (kind == BackendKind::Vulkan &&
            !LoadModuleFromFolder(backend, folder, L"ggml-vulkan.dll", true))
        {
            UnloadBackend(backend);
            return false;
        }

        if (!LoadModuleFromFolder(backend, folder, L"whisper.dll", true))
        {
            UnloadBackend(backend);
            return false;
        }

        backend.WhisperModule = backend.Modules[backend.ModuleCount - 1];
        if (!ResolveBackendFunctions(backend))
        {
            UnloadBackend(backend);
            return false;
        }

        return true;
    }

    void ActivateBackend(BackendApi & backend)
    {
        g_activeBackend = backend;
        backend = BackendApi{};

        switch (g_activeBackend.Kind)
        {
            case BackendKind::Cpu:
                g_lastBackendName = "cpu";
                break;
            case BackendKind::Vulkan:
                g_lastBackendName = "vulkan";
                break;
            default:
                g_lastBackendName.clear();
                break;
        }
    }

    template <typename TInit>
    whisper_context * TryInitializeBackend(BackendApi & backend, TInit & init)
    {
        try
        {
            return init(backend);
        }
        catch (const std::exception & exception)
        {
            SetLastBackendError(exception.what());
            return nullptr;
        }
        catch (...)
        {
            SetLastBackendError("Unhandled native exception during Whisper backend initialization.");
            return nullptr;
        }
    }

    template <typename TInit>
    whisper_context * InitializeContext(bool useGpu, TInit init)
    {
        if (g_activeBackend.WhisperModule != nullptr)
        {
            ClearLastBackendError();
            return TryInitializeBackend(g_activeBackend, init);
        }

        std::string gpuError;
        if (useGpu)
        {
            BackendApi gpuBackend;
            if (LoadBackend(gpuBackend, BackendKind::Vulkan))
            {
                if (auto * ctx = TryInitializeBackend(gpuBackend, init))
                {
                    ActivateBackend(gpuBackend);
                    return ctx;
                }

                gpuError = g_lastBackendError;
            }
            else
            {
                gpuError = g_lastBackendError;
            }

            UnloadBackend(gpuBackend);
        }

        BackendApi cpuBackend;
        if (!LoadBackend(cpuBackend, BackendKind::Cpu))
            return nullptr;

        if (auto * ctx = TryInitializeBackend(cpuBackend, init))
        {
            ActivateBackend(cpuBackend);
            if (!gpuError.empty())
                g_lastBackendError = gpuError;
            return ctx;
        }

        const auto cpuError = g_lastBackendError;
        if (!gpuError.empty() && !cpuError.empty())
        {
            g_lastBackendError = "GPU init failed: " + gpuError + " | CPU fallback failed: " + cpuError;
        }

        UnloadBackend(cpuBackend);
        return nullptr;
    }

    whisper_context_params GetDefaultContextParams()
    {
        if (g_activeBackend.ContextDefaultParams != nullptr)
            return g_activeBackend.ContextDefaultParams();

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return whisper_context_params{};

        whisper_context_params params{};
        try
        {
            params = backend.ContextDefaultParams();
        }
        catch (...)
        {
            SetLastBackendError("Failed to query default whisper context params.");
        }

        UnloadBackend(backend);
        return params;
    }

    whisper_full_params GetDefaultFullParams(whisper_sampling_strategy strategy)
    {
        if (g_activeBackend.FullDefaultParams != nullptr)
            return g_activeBackend.FullDefaultParams(strategy);

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return whisper_full_params{};

        whisper_full_params params{};
        try
        {
            params = backend.FullDefaultParams(strategy);
        }
        catch (...)
        {
            SetLastBackendError("Failed to query default whisper full params.");
        }

        UnloadBackend(backend);
        return params;
    }
}

BOOL APIENTRY DllMain(HMODULE moduleHandle, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_selfModule = moduleHandle;
        DisableThreadLibraryCalls(moduleHandle);
    }

    return TRUE;
}

extern "C"
{
    __declspec(dllexport) whisper_context * whisper_init_from_file_with_params(
        const char * path_model,
        whisper_context_params params)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return InitializeContext(
            params.use_gpu,
            [&](BackendApi & backend)
            {
                return backend.InitFromFileWithParams(path_model, params);
            });
    }

    __declspec(dllexport) whisper_context * whisper_init_from_buffer_with_params(
        void * buffer,
        size_t buffer_size,
        whisper_context_params params)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return InitializeContext(
            params.use_gpu,
            [&](BackendApi & backend)
            {
                return backend.InitFromBufferWithParams(buffer, buffer_size, params);
            });
    }

    __declspec(dllexport) const char * whisper_version()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_activeBackend.Version != nullptr)
            return CopyScratch(g_activeBackend.Version());

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return nullptr;

        const auto * version = backend.Version();
        const auto * copy = CopyScratch(version);
        UnloadBackend(backend);
        return copy;
    }

    __declspec(dllexport) int whisper_lang_max_id()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_activeBackend.LangMaxId != nullptr)
            return g_activeBackend.LangMaxId();

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return -1;

        const auto result = backend.LangMaxId();
        UnloadBackend(backend);
        return result;
    }

    __declspec(dllexport) int whisper_lang_id(const char * lang)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_activeBackend.LangId != nullptr)
            return g_activeBackend.LangId(lang);

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return -1;

        const auto result = backend.LangId(lang);
        UnloadBackend(backend);
        return result;
    }

    __declspec(dllexport) const char * whisper_lang_str(int id)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_activeBackend.LangStr != nullptr)
            return CopyScratch(g_activeBackend.LangStr(id));

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return nullptr;

        const auto * value = backend.LangStr(id);
        const auto * copy = CopyScratch(value);
        UnloadBackend(backend);
        return copy;
    }

    __declspec(dllexport) whisper_token whisper_token_eot(whisper_context * ctx)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.TokenEot(ctx);
    }

    __declspec(dllexport) const char * whisper_print_system_info()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_activeBackend.PrintSystemInfo != nullptr)
            return CopyScratch(g_activeBackend.PrintSystemInfo());

        BackendApi backend;
        if (!LoadBackend(backend, BackendKind::Cpu))
            return nullptr;

        const auto * info = backend.PrintSystemInfo();
        const auto * copy = CopyScratch(info);
        UnloadBackend(backend);
        return copy;
    }

    __declspec(dllexport) whisper_full_params whisper_full_default_params(whisper_sampling_strategy strategy)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return GetDefaultFullParams(strategy);
    }

    __declspec(dllexport) whisper_context_params whisper_context_default_params()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return GetDefaultContextParams();
    }

    __declspec(dllexport) whisper_full_params * whisper_full_default_params_by_ref(whisper_sampling_strategy strategy)
    {
        auto * params = new whisper_full_params;
        *params = whisper_full_default_params(strategy);
        return params;
    }

    __declspec(dllexport) whisper_context_params * whisper_context_default_params_by_ref()
    {
        auto * params = new whisper_context_params;
        *params = whisper_context_default_params();
        return params;
    }

    __declspec(dllexport) void whisper_free_params(whisper_full_params * params)
    {
        delete params;
    }

    __declspec(dllexport) void whisper_free_context_params(whisper_context_params * params)
    {
        delete params;
    }

    __declspec(dllexport) int whisper_full(whisper_context * ctx, whisper_full_params params, const float * samples, int n_samples)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.Full(ctx, params, samples, n_samples);
    }

    __declspec(dllexport) int whisper_full_n_segments(whisper_context * ctx)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullNSegments(ctx);
    }

    __declspec(dllexport) int whisper_full_lang_id(whisper_context * ctx)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullLangId(ctx);
    }

    __declspec(dllexport) int whisper_is_multilingual(whisper_context * ctx)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.IsMultilingual(ctx);
    }

    __declspec(dllexport) int64_t whisper_full_get_segment_t0(whisper_context * ctx, int i_segment)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullGetSegmentT0(ctx, i_segment);
    }

    __declspec(dllexport) int64_t whisper_full_get_segment_t1(whisper_context * ctx, int i_segment)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullGetSegmentT1(ctx, i_segment);
    }

    __declspec(dllexport) const char * whisper_full_get_segment_text(whisper_context * ctx, int i_segment)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullGetSegmentText(ctx, i_segment);
    }

    __declspec(dllexport) int whisper_full_n_tokens(whisper_context * ctx, int i_segment)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullNTokens(ctx, i_segment);
    }

    __declspec(dllexport) const char * whisper_full_get_token_text(whisper_context * ctx, int i_segment, int i_token)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullGetTokenText(ctx, i_segment, i_token);
    }

    __declspec(dllexport) whisper_token_data whisper_full_get_token_data(whisper_context * ctx, int i_segment, int i_token)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        return g_activeBackend.FullGetTokenData(ctx, i_segment, i_token);
    }

    __declspec(dllexport) void whisper_free(whisper_context * ctx)
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        g_activeBackend.Free(ctx);
    }

    __declspec(dllexport) const char * whisper_unity_get_active_backend()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_lastBackendName.empty())
            return nullptr;

        return g_lastBackendName.c_str();
    }

    __declspec(dllexport) const char * whisper_unity_get_last_error()
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_lastBackendError.empty())
            return nullptr;

        return g_lastBackendError.c_str();
    }
}
