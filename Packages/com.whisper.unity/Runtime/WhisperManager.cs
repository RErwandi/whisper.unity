using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Localization;
using UnityEngine.Localization.Settings;
using Whisper.Native;
using Whisper.Utils;

namespace Whisper
{
    /// <summary>
    /// Manages Whisper model lifecycle in Unity scene.
    /// </summary>
    public class WhisperManager : MonoBehaviour
    {
        [Tooltip("Log level for whisper loading and inference")]
        public LogLevel logLevel = LogLevel.Log;
        
        [Header("Model")]
        [SerializeField] 
        [Tooltip("Path to model weights file")]
        private string modelPath = "Whisper/ggml-tiny.bin";

        [SerializeField]
        [Tooltip("Optional English-specific model weights file used when the current language is English")]
        private string englishModelPath;

        [SerializeField]
        [Tooltip("Optional list of player-selectable Whisper models")]
        private List<WhisperModelDefinition> availableModels = new List<WhisperModelDefinition>();

        [SerializeField]
        [Tooltip("Model index used by InitModel when availableModels is populated")]
        private int defaultModelIndex;
        
        [SerializeField]
        [Tooltip("Determines whether the StreamingAssets folder should be prepended to the model path")]
        private bool isModelPathInStreamingAssets = true;

        [SerializeField]
        [Tooltip("Determines whether the StreamingAssets folder should be prepended to the English model path")]
        private bool isEnglishModelPathInStreamingAssets = true;
        
        [SerializeField] 
        [Tooltip("Should model weights be loaded on awake?")]
        private bool initOnAwake = true;
        
        [Header("Inference")]
        [Tooltip("Try to load whisper in GPU for faster inference")]
        [SerializeField]
        private bool useGpu;

        [Tooltip("GPU device index used by native whisper.cpp backends")]
        [SerializeField]
        private int gpuDevice;

        [Tooltip("Allow WHISPER_ARG_DEVICE or GPU_DEVICE environment variable to override GPU Device")]
        [SerializeField]
        private bool gpuDeviceEnvOverride = true;
        
        [Tooltip("Use the Flash Attention algorithm for faster inference")]
        [SerializeField]
        private bool flashAttention;

        [Header("Language")] 
        [Tooltip("Output text language. Use empty or \"auto\" for auto-detection.")]
        public string language = "en";

        [Tooltip("Force output text to English translation. Improves translation quality.")]
        public bool translateToEnglish;

        [Header("Advanced settings")] 
        [SerializeField]
        private WhisperSamplingStrategy strategy = WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY;

        [Tooltip("Do not use past transcription (if any) as initial prompt for the decoder.")]
        public bool noContext = true;

        [Tooltip("Force single segment output (useful for streaming).")]
        public bool singleSegment;

        [Tooltip("Output tokens with their confidence in each segment.")]
        public bool enableTokens;

        [Tooltip("Initial prompt as a string variable. " +
                 "It should improve transcription quality or guide it to the right direction.")]
        [TextArea]
        public string initialPrompt;

        [Header("Streaming settings")] 
        [Tooltip("Minimal portions of audio that will be processed by whisper stream in seconds.")]
        public float stepSec = 3f;

        [Tooltip("How many seconds of previous segment will be used for current segment.")]
        public float keepSec = 0.2f;

        [Tooltip("How many seconds of audio will be recurrently transcribe until context update.")]
        public float lengthSec = 10f;
        
        [Tooltip("Should stream modify whisper prompt for better context handling?")]
        public bool updatePrompt = true;

        [Tooltip("If false stream will use all information from previous iteration.")]
        public bool dropOldBuffer;

        [Tooltip("If true stream will ignore audio chunks with no detected speech.")]
        public bool useVad = true;

        [Header("Experimental settings")]
        [Tooltip("[EXPERIMENTAL] Output timestamps for each token. Need enabled tokens to work.")]
        public bool tokensTimestamps;

        [Tooltip("[EXPERIMENTAL] Overwrite the audio context size (0 = use default). " +
                 "These can significantly reduce the quality of the output.")]
        public int audioCtx;

        /// <summary>
        /// Raised when whisper transcribed a new text segment from audio. 
        /// </summary>
        public event OnNewSegmentDelegate OnNewSegment;
        
        /// <summary>
        /// Raised when whisper made some progress in transcribing audio.
        /// Progress changes from 0 to 100 included.
        /// </summary>
        public event OnProgressDelegate OnProgress;

        private WhisperWrapper _whisper;
        private WhisperParams _params;
        private readonly MainThreadDispatcher _dispatcher = new MainThreadDispatcher();
        private int _activeModelIndex;
        private string _loadedModelPath;
        private bool _isLoadedModelPathInStreamingAssets;
        private const string SpeechModelPrefsKey = "speech_model";

        public string ModelPath
        {
            get => modelPath;
            set
            {
                if (IsLoaded || IsLoading)
                {
                    throw new InvalidOperationException("Cannot change model path after loading the model");
                }

                modelPath = value;
            }
        }

        public IReadOnlyList<WhisperModelDefinition> AvailableModels => availableModels;
        public int ActiveModelIndex => _activeModelIndex;
        public string ActiveBackend => WhisperWrapper.TryGetActiveBackend(out var backend) ? backend : null;
        public string LastBackendError => WhisperWrapper.TryGetLastBackendError(out var error) ? error : null;
        public bool UseGpu => useGpu;
        public int GpuDevice => gpuDevice;
        public bool GpuDeviceEnvOverride => gpuDeviceEnvOverride;
        public bool FlashAttention => flashAttention;
        
        public bool IsModelPathInStreamingAssets
        {
            get => isModelPathInStreamingAssets;
            set
            {
                if (IsLoaded || IsLoading)
                {
                    throw new InvalidOperationException("Cannot change model path after loading the model");
                }

                isModelPathInStreamingAssets = value;
            }
        }
        
        /// <summary>
        /// Checks if whisper weights are loaded and ready to be used.
        /// </summary>
        public bool IsLoaded => _whisper != null;

        /// <summary>
        /// Checks if whisper weights are still loading and not ready.
        /// </summary>
        public bool IsLoading { get; private set; }

        private async void Awake()
        {
            LogUtils.Level = logLevel;
            OnLocaleChanged(LocalizationSettings.SelectedLocale);
            LocalizationSettings.SelectedLocaleChanged += OnLocaleChanged;
            
            if (!initOnAwake)
                return;
            await InitModel();
        }

        private void OnDestroy()
        {
            LocalizationSettings.SelectedLocaleChanged -= OnLocaleChanged;
            UnloadModel();
        }

        private async void OnLocaleChanged(Locale locale)
        {
            language = GetWhisperLangCode(locale);
            await ReloadModelForLocaleIfNeeded();
        }
        
        private string GetWhisperLangCode(Locale locale)
        {
            if (locale == null)
                return "en"; // fallback to English

            var code = locale.Identifier.Code;

            // Whisper expects ISO 639-1 language code (2 letters)
            // If regional variant exists, just take first 2 letters
            if (code.Contains("-"))
            {
                var parts = code.Split('-');
                return parts[0]; 
            }

            return code;
        }
        

        private void OnValidate()
        {
            LogUtils.Level = logLevel;

            if (defaultModelIndex < 0)
                defaultModelIndex = 0;

            if (gpuDevice < 0)
                gpuDevice = 0;

            if (availableModels.Count > 0)
            {
                var maxIndex = availableModels.Count - 1;
                if (defaultModelIndex > maxIndex)
                    defaultModelIndex = maxIndex;
            }
        }

        private void Update()
        {
            _dispatcher.Update();
        }

        /// <summary>
        /// Load model and default parameters. Prepare it for text transcription.
        /// </summary>
        public async Task InitModel()
        {
            if (availableModels.Count > 0)
            {
                var startupModelIndex = GetStartupModelIndex();
                await LoadModelAsync(startupModelIndex);
                return;
            }

            var resolvedModel = ResolveModelTarget(
                modelPath,
                isModelPathInStreamingAssets,
                englishModelPath,
                isEnglishModelPathInStreamingAssets);
            await LoadModelAsync(
                resolvedModel.ModelPath,
                resolvedModel.IsPathInStreamingAssets);
        }

        /// <summary>
        /// Load one of the configured Whisper models by index.
        /// </summary>
        public async Task<bool> LoadModelAsync(int modelIndex)
        {
            if (!TryGetModelDefinition(modelIndex, out var model))
            {
                LogUtils.Error($"Whisper model index {modelIndex} is invalid.");
                return false;
            }

            var resolvedModel = ResolveModelTarget(model);
            if (IsLoaded &&
                _activeModelIndex == modelIndex &&
                IsResolvedModelLoaded(resolvedModel))
            {
                return true;
            }

            var loaded = await LoadModelAsync(
                resolvedModel.ModelPath,
                resolvedModel.IsPathInStreamingAssets);
            if (loaded)
            {
                _activeModelIndex = modelIndex;
            }

            return loaded;
        }

        /// <summary>
        /// Load Whisper model from explicit path and unload current model if necessary.
        /// </summary>
        public async Task<bool> LoadModelAsync(string targetModelPath, bool isPathInStreamingAssets)
        {
            // check if model is already loaded or actively loading
            if (IsLoading)
            {
                LogUtils.Warning("Whisper model is already loading!");
                return false;
            }

            if (string.IsNullOrWhiteSpace(targetModelPath))
            {
                LogUtils.Error("Whisper model path is empty!");
                return false;
            }

            if (IsLoaded &&
                string.Equals(_loadedModelPath, targetModelPath, StringComparison.Ordinal) &&
                _isLoadedModelPathInStreamingAssets == isPathInStreamingAssets)
            {
                return true;
            }

            IsLoading = true;
            try
            {
                UnloadModel();

                var path = isPathInStreamingAssets
                    ? Path.Combine(Application.streamingAssetsPath, targetModelPath)
                    : targetModelPath;

                var context = CreateContextParams();
                _whisper = await WhisperWrapper.InitFromFileAsync(path, context);
                if (_whisper == null)
                {
                    LogUtils.Error($"Failed to initialize Whisper model from {path}.");
                    return false;
                }

                _params = WhisperParams.GetDefaultParams(strategy);
                UpdateParams();
                
                _whisper.OnNewSegment += OnNewSegmentHandler;
                _whisper.OnProgress += OnProgressHandler;
                _loadedModelPath = targetModelPath;
                _isLoadedModelPathInStreamingAssets = isPathInStreamingAssets;
                LogBackendSelection(path, context);
                Debug.Log($"[Whisper] Initialized model: {path}");
                return true;
            }
            catch (Exception e)
            {
                LogUtils.Exception(e);
                return false;
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Switch currently active model to one of the configured options.
        /// </summary>
        public async Task<bool> SwitchModelAsync(int modelIndex)
        {
            return await LoadModelAsync(modelIndex);
        }

        public bool SetGpuOptions(bool enableGpu, bool enableFlashAttention)
        {
            return SetGpuOptions(enableGpu, enableFlashAttention, gpuDevice, gpuDeviceEnvOverride);
        }

        public bool SetGpuOptions(bool enableGpu, bool enableFlashAttention, int gpuDeviceIndex,
            bool allowEnvironmentOverride = true)
        {
            if (IsLoading)
            {
                LogUtils.Warning("Cannot change Whisper GPU options while model is loading.");
                return false;
            }

            if (gpuDeviceIndex < 0)
            {
                LogUtils.Warning("Cannot set Whisper GPU device to a negative index.");
                return false;
            }

            useGpu = enableGpu;
            flashAttention = enableFlashAttention;
            gpuDevice = gpuDeviceIndex;
            gpuDeviceEnvOverride = allowEnvironmentOverride;
            return true;
        }

        /// <summary>
        /// Release currently loaded model from memory.
        /// </summary>
        public void UnloadModel()
        {
            if (_whisper == null)
                return;

            _whisper.OnNewSegment -= OnNewSegmentHandler;
            _whisper.OnProgress -= OnProgressHandler;
            _whisper.Dispose();
            _whisper = null;
            _params = null;
            _activeModelIndex = -1;
            _loadedModelPath = null;
            _isLoadedModelPathInStreamingAssets = false;
        }
        
        /// <summary>
        /// Checks if currently loaded whisper model supports multilingual transcription.
        /// </summary>
        public bool IsMultilingual()
        {
            if (!IsLoaded)
            {
                LogUtils.Error("Whisper model isn't loaded! Init Whisper model first!");
                return false;
            }

            return _whisper.IsMultilingual;
        }

        /// <summary>
        /// Start async transcription of audio clip.
        /// </summary>
        /// <returns>Full audio transcript. Null if transcription failed.</returns>
        public async Task<WhisperResult> GetTextAsync(AudioClip clip)
        {
            await EnsureDefaultModelLoaded();
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
                return null;

            UpdateParams();
            var res = await _whisper.GetTextAsync(clip, _params);
            return res;
        }
        
        /// <summary>
        /// Start async transcription of audio buffer.
        /// </summary>
        /// <param name="samples">Raw audio buffer.</param>
        /// <param name="frequency">Audio sample rate.</param>
        /// <param name="channels">Audio channels count.</param>
        /// <returns>Full audio transcript. Null if transcription failed.</returns>
        public async Task<WhisperResult> GetTextAsync(float[] samples, int frequency, int channels)
        {
            await EnsureDefaultModelLoaded();
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
                return null;

            UpdateParams();
            var res = await _whisper.GetTextAsync(samples, frequency, channels, _params);
            return res;
        }
        
        /// <summary>
        /// Create a new instance of Whisper streaming transcription.
        /// </summary>
        /// <param name="frequency">Audio sample rate.</param>
        /// <param name="channels">Audio channels count.</param>
        /// <returns>New streaming transcription. Null if failed.</returns>
        public async Task<WhisperStream> CreateStream(int frequency, int channels)
        {
            await EnsureDefaultModelLoaded();
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
            {
                LogUtils.Error("Model weights aren't loaded! Load model first!");
                return null;
            }

            var param = new WhisperStreamParams(_params,
                frequency, channels, stepSec, keepSec, lengthSec, updatePrompt,
                dropOldBuffer, useVad);
            var stream = new WhisperStream(_whisper, param);
            return stream;
        }
        
        /// <summary>
        /// Create a new instance of Whisper streaming transcription from microphone input.
        /// </summary>
        /// <returns>New streaming transcription. Null if failed.</returns>
        public async Task<WhisperStream> CreateStream(MicrophoneRecord microphone)
        {
            await EnsureDefaultModelLoaded();
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
            {
                LogUtils.Error("Model weights aren't loaded! Load model first!");
                return null;
            }
            
            // TODO: unity support only single input channel for microphone
            var channels = 1;
            var frequency = microphone.frequency;
            var param = new WhisperStreamParams(_params,
                frequency, channels, stepSec, keepSec, lengthSec, updatePrompt,
                dropOldBuffer, useVad);
            var stream = new WhisperStream(_whisper, param, microphone);
            return stream;
        }
        
        private void UpdateParams()
        {
            _params.Language = language;
            _params.Translate = translateToEnglish;
            _params.NoContext = noContext;
            _params.SingleSegment = singleSegment;
            _params.AudioCtx = audioCtx;
            _params.EnableTokens = enableTokens;
            _params.TokenTimestamps = tokensTimestamps;
            _params.InitialPrompt = initialPrompt;
        }
        
        private WhisperContextParams CreateContextParams()
        {
            var context = WhisperContextParams.GetDefaultParams();
            context.UseGpu = useGpu;
            context.FlashAttn = flashAttention;
            context.GpuDevice = ResolveGpuDevice(useGpu, gpuDevice, gpuDeviceEnvOverride);
            return context;
        }

        public static int ResolveGpuDevice(bool useGpu, int configuredGpuDevice, bool useEnvironmentOverride = true)
        {
            if (!useGpu)
                return 0;

            if (configuredGpuDevice < 0)
            {
                LogUtils.Warning($"Configured GPU device {configuredGpuDevice} is invalid. Falling back to device 0.");
                configuredGpuDevice = 0;
            }

            if (!useEnvironmentOverride)
                return configuredGpuDevice;

            if (TryGetGpuDeviceFromEnvironment("WHISPER_ARG_DEVICE", out var whisperArgDevice))
                return whisperArgDevice;

            if (TryGetGpuDeviceFromEnvironment("GPU_DEVICE", out var gpuDeviceEnv))
                return gpuDeviceEnv;

            return configuredGpuDevice;
        }

        private static bool TryGetGpuDeviceFromEnvironment(string variableName, out int device)
        {
            device = 0;
            var value = Environment.GetEnvironmentVariable(variableName);
            if (string.IsNullOrWhiteSpace(value))
                return false;

            if (int.TryParse(value, out device) && device >= 0)
                return true;

            LogUtils.Warning($"Ignoring invalid {variableName} value '{value}'. GPU device should be a non-negative integer.");
            device = 0;
            return false;
        }

        private static string GetEnvironmentValueForLog(string variableName)
        {
            var value = Environment.GetEnvironmentVariable(variableName);
            return string.IsNullOrWhiteSpace(value) ? "<not set>" : value;
        }

        private void LogBackendSelection(string path, WhisperContextParams context)
        {
            LogUtils.Log($"Whisper GPU request: useGpu={context.UseGpu}, gpuDevice={context.GpuDevice}, " +
                         $"WHISPER_ARG_DEVICE={GetEnvironmentValueForLog("WHISPER_ARG_DEVICE")}, " +
                         $"GPU_DEVICE={GetEnvironmentValueForLog("GPU_DEVICE")}");

            if (!WhisperWrapper.TryGetActiveBackend(out var backend))
                return;

            if (string.Equals(backend, "cpu", StringComparison.OrdinalIgnoreCase) &&
                useGpu &&
                WhisperWrapper.TryGetLastBackendError(out var error))
            {
                LogUtils.Warning(
                    $"Whisper GPU initialization failed for model '{path}'. Falling back to CPU backend. Reason: {error}");
                return;
            }

            LogUtils.Log($"Whisper backend selected: {backend}");
        }

        private async Task<bool> CheckIfLoaded()
        {
            if (!IsLoaded && !IsLoading)
            {
                LogUtils.Error("Whisper model isn't loaded! Init Whisper model first!");
                return false;
            }

            // wait while model still loading
            while (IsLoading)
            {
                await Task.Yield();
            }

            return IsLoaded;
        }

        private bool TryGetModelDefinition(int modelIndex, out WhisperModelDefinition model)
        {
            if (modelIndex >= 0 && modelIndex < availableModels.Count)
            {
                model = availableModels[modelIndex];
                return model != null;
            }

            model = null;
            return false;
        }

        private async Task EnsureDefaultModelLoaded()
        {
            if (IsLoaded)
                return;

            await InitModel();
        }

        private int GetStartupModelIndex()
        {
            var savedModelIndex = PlayerPrefs.GetInt(SpeechModelPrefsKey, defaultModelIndex);
            return Mathf.Clamp(savedModelIndex, 0, availableModels.Count - 1);
        }

        private async Task ReloadModelForLocaleIfNeeded()
        {
            if (!IsLoaded && !IsLoading)
                return;

            while (IsLoading)
            {
                await Task.Yield();
            }

            if (!IsLoaded)
                return;

            if (availableModels.Count > 0)
            {
                if (!TryGetModelDefinition(_activeModelIndex, out var model))
                    return;

                var resolvedModel = ResolveModelTarget(model);
                if (IsResolvedModelLoaded(resolvedModel))
                    return;

                await LoadModelAsync(_activeModelIndex);
                return;
            }

            var resolvedSingleModel = ResolveModelTarget(
                modelPath,
                isModelPathInStreamingAssets,
                englishModelPath,
                isEnglishModelPathInStreamingAssets);
            if (IsResolvedModelLoaded(resolvedSingleModel))
                return;

            await LoadModelAsync(
                resolvedSingleModel.ModelPath,
                resolvedSingleModel.IsPathInStreamingAssets);
        }

        private bool IsResolvedModelLoaded(ResolvedModelTarget resolvedModel)
        {
            return string.Equals(_loadedModelPath, resolvedModel.ModelPath, StringComparison.Ordinal) &&
                   _isLoadedModelPathInStreamingAssets == resolvedModel.IsPathInStreamingAssets;
        }

        private ResolvedModelTarget ResolveModelTarget(WhisperModelDefinition model)
        {
            return ResolveModelTarget(
                model.ModelPath,
                model.IsPathInStreamingAssets,
                model.EnglishModelPath,
                model.IsEnglishPathInStreamingAssets);
        }

        private ResolvedModelTarget ResolveModelTarget(
            string defaultModelPath,
            bool isDefaultModelPathInStreamingAssets,
            string englishOverrideModelPath,
            bool isEnglishOverrideModelPathInStreamingAssets)
        {
            var useEnglishModel = string.Equals(language, "en", StringComparison.Ordinal) &&
                                  !string.IsNullOrWhiteSpace(englishOverrideModelPath);
            if (useEnglishModel)
            {
                return new ResolvedModelTarget(
                    englishOverrideModelPath,
                    isEnglishOverrideModelPathInStreamingAssets);
            }

            return new ResolvedModelTarget(defaultModelPath, isDefaultModelPathInStreamingAssets);
        }
        
        private void OnNewSegmentHandler(WhisperSegment segment)
        {
            _dispatcher.Execute(() =>
            {
                OnNewSegment?.Invoke(segment);
            });
        }
        
        private void OnProgressHandler(int progress)
        {
            _dispatcher.Execute(() =>
            {
                OnProgress?.Invoke(progress);
            });
        }

        private readonly struct ResolvedModelTarget
        {
            public readonly string ModelPath;
            public readonly bool IsPathInStreamingAssets;

            public ResolvedModelTarget(string modelPath, bool isPathInStreamingAssets)
            {
                ModelPath = modelPath;
                IsPathInStreamingAssets = isPathInStreamingAssets;
            }
        }
    }
}
