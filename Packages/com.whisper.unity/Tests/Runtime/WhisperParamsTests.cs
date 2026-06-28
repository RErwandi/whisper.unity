using System;
using NUnit.Framework;
using Whisper.Native;

namespace Whisper.Tests
{
    [TestFixture]
    public class WhisperParamsTests
    {
        [Test]
        [TestCase(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY)]
        [TestCase(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH)]
        public void DefaultParamsStrategyTest(WhisperSamplingStrategy strategy)
        {
            var param = WhisperParams.GetDefaultParams(strategy);
            Assert.NotNull(param);
            Assert.AreEqual(param.Strategy, strategy);
        }

        [Test]
        public void LanguageParamsTest()
        {
            var param = WhisperParams.GetDefaultParams();
            Assert.NotNull(param);

            // check default language
            Assert.AreEqual("en", param.Language);
            
            // check auto language
            param.Language = "";
            Assert.AreEqual("", param.Language);
            param.Language = null;
            Assert.AreEqual(null, param.Language);
            
            // check language switch
            param.Language = "de";
            Assert.AreEqual("de", param.Language);
        }

        [Test]
        public void PromptParamsTest()
        {
            var param = WhisperParams.GetDefaultParams();
            Assert.NotNull(param);
            
            // check get default prompt
            Assert.DoesNotThrow(() => { var tmp = param.InitialPrompt; });
            
            // check no prompt provided
            param.InitialPrompt = "";
            Assert.AreEqual("", param.InitialPrompt);
            param.InitialPrompt = null;
            Assert.AreEqual(null, param.InitialPrompt);
            
            // check prompt changing
            const string constPrompt = "hello how is it going always use lowercase no punctuation goodbye one two three start stop i you me they";
            param.InitialPrompt = constPrompt;
            Assert.AreEqual(constPrompt, param.InitialPrompt);
        }

        [Test]
        public void ContextGpuDeviceParamsTest()
        {
            var param = WhisperContextParams.GetDefaultParams();
            Assert.NotNull(param);

            Assert.AreEqual(0, param.GpuDevice);
            param.GpuDevice = 1;
            Assert.AreEqual(1, param.GpuDevice);
            Assert.Throws<ArgumentException>(() => param.GpuDevice = -1);
        }

        [Test]
        public void ResolveGpuDeviceUsesConfiguredValueWithoutEnvironment()
        {
            WithGpuDeviceEnvironment(null, null, () =>
            {
                Assert.AreEqual(2, WhisperManager.ResolveGpuDevice(true, 2));
            });
        }

        [Test]
        public void ResolveGpuDeviceReturnsZeroWhenGpuDisabled()
        {
            WithGpuDeviceEnvironment("3", "2", () =>
            {
                Assert.AreEqual(0, WhisperManager.ResolveGpuDevice(false, 1));
            });
        }

        [Test]
        public void ResolveGpuDeviceUsesWhisperArgDeviceEnvironment()
        {
            WithGpuDeviceEnvironment("1", null, () =>
            {
                Assert.AreEqual(1, WhisperManager.ResolveGpuDevice(true, 0));
            });
        }

        [Test]
        public void ResolveGpuDeviceUsesGpuDeviceEnvironment()
        {
            WithGpuDeviceEnvironment(null, "3", () =>
            {
                Assert.AreEqual(3, WhisperManager.ResolveGpuDevice(true, 0));
            });
        }

        [Test]
        public void ResolveGpuDeviceFallsBackOnInvalidEnvironment()
        {
            WithGpuDeviceEnvironment("invalid", "-1", () =>
            {
                Assert.AreEqual(2, WhisperManager.ResolveGpuDevice(true, 2));
            });
        }

        private static void WithGpuDeviceEnvironment(string whisperArgDevice, string gpuDevice, Action action)
        {
            var oldWhisperArgDevice = Environment.GetEnvironmentVariable("WHISPER_ARG_DEVICE");
            var oldGpuDevice = Environment.GetEnvironmentVariable("GPU_DEVICE");
            try
            {
                Environment.SetEnvironmentVariable("WHISPER_ARG_DEVICE", whisperArgDevice);
                Environment.SetEnvironmentVariable("GPU_DEVICE", gpuDevice);
                action();
            }
            finally
            {
                Environment.SetEnvironmentVariable("WHISPER_ARG_DEVICE", oldWhisperArgDevice);
                Environment.SetEnvironmentVariable("GPU_DEVICE", oldGpuDevice);
            }
        }
    }
}
