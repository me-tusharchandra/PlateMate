using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Networking;
using TMPro;
using System.Collections;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

public class QuestIntegration : MonoBehaviour
{
    [SerializeField] private InputActionReference leftGrabAction;
    [SerializeField] private TextMeshProUGUI logText;

    [Header("Server Settings")]
    [SerializeField] private string serverUrl = "http://10.47.1.123:5000"; // Your PC's IP
    [SerializeField] private string promptText = "Based on what you can see in this image, analyze if this food product is safe for consumption with my allergies and health conditions. Keep your response under 3 lines.";

    [Header("Azure Settings")]
    [SerializeField] private string subscriptionKey = "dc20c430e11643b3a192c8a22c9df3ce";
    [SerializeField] private string region = "centralindia";

    // List of potential server URLs to try
    private List<string> serverUrls = new List<string>();
    private string activeServerUrl;
    private bool isConnected = false;
   
    // Audio recording variables
    private string deviceName;
    private const int CLIP_FREQUENCY = 16000;
    private const int RECORDING_LENGTH = 20; // Maximum recording length in seconds
    private bool isRecording = false;
    private AudioClip recordingClip;
    private bool isProcessing = false;
    private SpeechSynthesizer synthesizer;
    private bool isSynthesizing = false;
    private bool isQuitting = false;
    private string transcribedText = "";

    private void OnEnable() => leftGrabAction.action.Enable();
    private void OnDisable() => leftGrabAction.action.Disable();

    private void Start()
    {
        // Add potential server URLs to try
        serverUrls.Add(serverUrl); // User-provided URL
        serverUrls.Add("http://127.0.0.1:5000"); // Localhost
        serverUrls.Add("http://10.0.2.2:5000"); // Android emulator special IP for localhost

        // Try to find PC's IP automatically in common subnets
        string[] commonSubnets = { "192.168.1.", "192.168.0.", "10.0.0." };
        foreach (string subnet in commonSubnets)
        {
            for (int i = 1; i <= 10; i++) // Try first 10 IPs in subnet
            {
                if (!serverUrls.Contains($"http://{subnet}{i}:5000"))
                {
                    serverUrls.Add($"http://{subnet}{i}:5000");
                }
            }
        }

        // Initialize microphone and speech services
        InitializeMicrophone();
        SetupSpeechServices();
       
        // Check server status on startup
        StartCoroutine(TryConnectToServer());
    }

    private void InitializeMicrophone()
    {
        if (Microphone.devices.Length > 0)
        {
            Debug.Log("Available microphones: " + string.Join(", ", Microphone.devices));
            deviceName = Microphone.devices[0];

            // Search for Quest microphone
            foreach (string device in Microphone.devices)
            {
                Debug.Log($"Found microphone: {device}");
                if (device.ToUpper().Contains("ANDROID") || device.ToUpper().Contains("OCULUS") || device.ToUpper().Contains("META"))
                {
                    deviceName = device;
                    break;
                }
            }

            Debug.Log($"Selected microphone: {deviceName}");
            UpdateLog($"Using mic: {deviceName}");
        }
        else
        {
            Debug.LogError("No microphones found!");
            UpdateLog("No microphone detected!");
        }
    }

    private void SetupSpeechServices()
    {
        var speechConfig = SpeechConfig.FromSubscription(subscriptionKey, region);
        speechConfig.SpeechRecognitionLanguage = "en-US";
        speechConfig.SpeechSynthesisVoiceName = "en-US-JennyNeural";
        synthesizer = new SpeechSynthesizer(speechConfig);
    }

    private IEnumerator TryConnectToServer()
    {
        UpdateLog("Searching for PlateMate server...");

        foreach (string url in serverUrls)
        {
            UpdateLog($"Trying: {url}");

            using (UnityWebRequest request = UnityWebRequest.Get($"{url}/status"))
            {
                request.timeout = 5; // 5 second timeout
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    activeServerUrl = url;
                    isConnected = true;
                    UpdateLog($"Connected to PlateMate server!");
                    yield break; // Exit once we find a working server
                }
            }

            yield return new WaitForSeconds(0.1f); // Small delay between attempts
        }

        if (!isConnected)
        {
            UpdateLog("Failed to connect to PlateMate server. Please check:\n" +
                      "1. Server is running\n" +
                      "2. PC and Quest are on same network\n" +
                      "3. Firewall allows connections");
        }
    }

    void Update()
    {
        // Start recording when left grab button is pressed
        if (leftGrabAction.action.WasPressedThisFrame())
        {
            if (isConnected && !isProcessing && !isRecording)
            {
                StartRecording();
            }
            else if (!isConnected)
            {
                StartCoroutine(TryConnectToServer());
            }
        }
       
        // Stop recording and process when left grab button is released
        if (leftGrabAction.action.WasReleasedThisFrame() && isRecording)
        {
            StopRecordingAndProcess();
        }
    }
   
    private void StartRecording()
    {
        UpdateLog("Recording... Ask about the food product");
       
        // Start recording using Unity's Microphone class
        recordingClip = Microphone.Start(deviceName, false, RECORDING_LENGTH, CLIP_FREQUENCY);
        isRecording = true;

        if (recordingClip == null)
        {
            Debug.LogError("Failed to start recording!");
            UpdateLog("Recording failed to start");
            isRecording = false;
        }
    }
   
    private void StopRecordingAndProcess()
    {
        if (!isRecording) return;
       
        isProcessing = true;
        StartCoroutine(ProcessRecordingAndCapture());
    }
   
    private IEnumerator ProcessRecordingAndCapture()
    {
        UpdateLog("Processing your speech...");
       
        // Stop recording
        Microphone.End(deviceName);
        isRecording = false;
       
        // Get the recorded data
        float[] audioData = new float[recordingClip.samples * recordingClip.channels];
        recordingClip.GetData(audioData, 0);

        // Convert to WAV
        byte[] wavFile = ConvertToWav(audioData, recordingClip.channels, recordingClip.frequency);
       
        // Start transcription process
        var transcriptionTask = TranscribeAudioAsync(wavFile);
       
        // Capture the screenshot while transcription is happening
        yield return new WaitForEndOfFrame();
        Texture2D screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenshot.Apply();
       
        // Wait for transcription to complete
        while (!transcriptionTask.IsCompleted)
        {
            yield return null;
        }
       
        transcribedText = transcriptionTask.Result;
       
        // If transcription succeeded, send to server
        if (!string.IsNullOrEmpty(transcribedText))
        {
            UpdateLog($"You asked: {transcribedText}");
           
            // Combine user's question with the base prompt
            string combinedPrompt = $"{transcribedText}. {promptText}";
           
            // Convert screenshot to base64
            byte[] jpgBytes = screenshot.EncodeToJPG(85);
            string base64Image = System.Convert.ToBase64String(jpgBytes);
           
            // Send to server with both the image and combined prompt
            yield return StartCoroutine(AnalyzeImage(base64Image, combinedPrompt));
        }
        else
        {
            UpdateLog("Couldn't understand your speech. Please try again.");
            isProcessing = false;
        }
       
        // Clean up the screenshot texture
        Destroy(screenshot);
    }
   
    private async System.Threading.Tasks.Task<string> TranscribeAudioAsync(byte[] wavFile)
    {
        var speechConfig = SpeechConfig.FromSubscription(subscriptionKey, region);
        string recognizedText = "";
        var taskCompletionSource = new System.Threading.Tasks.TaskCompletionSource<string>();
        SpeechRecognizer tempRecognizer = null;
       
        try
        {
            // Create a memory stream from the WAV file
            using (var memoryStream = new System.IO.MemoryStream(wavFile))
            {
                // Create audio stream from memory stream
                using (var audioInputStream = AudioInputStream.CreatePushStream())
                {
                    // Write WAV data to the push stream
                    byte[] buffer = new byte[32000];
                    int bytesRead;
                    while ((bytesRead = memoryStream.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        audioInputStream.Write(buffer, bytesRead);
                    }
               
                    // Create audio config from the stream
                    var audioConfig = AudioConfig.FromStreamInput(audioInputStream);
                   
                    tempRecognizer = new SpeechRecognizer(speechConfig, audioConfig);
                    tempRecognizer.Recognized += (s, e) =>
                    {
                        if (e.Result.Reason == ResultReason.RecognizedSpeech)
                        {
                            taskCompletionSource.TrySetResult(e.Result.Text);
                        }
                        else
                        {
                            taskCompletionSource.TrySetResult(string.Empty);
                        }
                    };

                    await tempRecognizer.StartContinuousRecognitionAsync();
                }
            }
           
            // Wait for recognition to complete
            recognizedText = await taskCompletionSource.Task;
           
            if (tempRecognizer != null)
            {
                await tempRecognizer.StopContinuousRecognitionAsync();
                tempRecognizer.Dispose();
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Speech recognition failed: {ex}");
            if (tempRecognizer != null)
            {
                tempRecognizer.Dispose();
            }
        }
       
        return recognizedText;
    }
   
    private byte[] ConvertToWav(float[] audioData, int channels, int frequency)
    {
        using (var memoryStream = new System.IO.MemoryStream())
        {
            using (var writer = new System.IO.BinaryWriter(memoryStream))
            {
                // WAV header
                writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
                writer.Write(0); // Final size
                writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
                writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
                writer.Write(16); // Subchunk1Size
                writer.Write((short)1); // AudioFormat (PCM)
                writer.Write((short)channels);
                writer.Write(frequency);
                writer.Write(frequency * channels * 2); // ByteRate
                writer.Write((short)(channels * 2)); // BlockAlign
                writer.Write((short)16); // BitsPerSample
                writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
                writer.Write(audioData.Length * 2);

                // Convert float audio data to 16-bit PCM
                foreach (float sample in audioData)
                {
                    writer.Write((short)(sample * 32767f));
                }

                // Update final size
                var size = (int)writer.BaseStream.Length - 8;
                writer.Seek(4, System.IO.SeekOrigin.Begin);
                writer.Write(size);
            }
            return memoryStream.ToArray();
        }
    }

    private IEnumerator AnalyzeImage(string base64Image, string customPrompt)
    {
        UpdateLog("Sending to PlateMate for analysis...");

        // Create the JSON data to send
        string jsonData = $"{{\"prompt\": \"{customPrompt}\", \"image\": \"{base64Image}\"}}";

        // Send the request to the Python server
        using (UnityWebRequest request = new UnityWebRequest($"{activeServerUrl}/analyze", "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                UpdateLog($"Error: {request.error}");
                isConnected = false; // Reset connection status on error
            }
            else
            {
                string response = request.downloadHandler.text;
                Debug.Log($"Raw response: {response}");

                // Parse the JSON response
                string analysisResult = ParseServerResponse(response);
               
                // Update UI with the analysis result
                UpdateLog(analysisResult);
               
                // Speak the result aloud
                SpeakResponse(CleanTextForSpeech(analysisResult));
            }
           
            isProcessing = false;
        }
    }

    private string ParseServerResponse(string jsonResponse)
    {
        try
        {
            // Parse JSON using JsonUtility
            ServerResponse response = JsonUtility.FromJson<ServerResponse>(jsonResponse);

            if (response != null && response.success)
            {
                return response.response;
            }
            else
            {
                // Try manual parsing if JsonUtility fails
                Match match = Regex.Match(jsonResponse, "\"response\":\\s*\"([^\"]+)\"");
                if (match.Success)
                {
                    string text = match.Groups[1].Value;
                    text = Regex.Unescape(text);
                    return text;
                }

                return "Failed to parse server response";
            }
        }
        catch (System.Exception e)
        {
            return $"Error parsing response: {e.Message}";
        }
    }
   
    private async void SpeakResponse(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            Debug.LogError("No text to synthesize");
            return;
        }

        try
        {
            isSynthesizing = true;
            Debug.Log($"Speaking: {text}");
           
            using (var result = await synthesizer.SpeakTextAsync(text))
            {
                if (result.Reason == ResultReason.SynthesizingAudioCompleted)
                {
                    Debug.Log("Speech synthesis completed successfully");
                }
                else
                {
                    Debug.LogError($"Speech synthesis failed: {result.Reason}");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error in speech synthesis: {e.Message}");
        }
        finally
        {
            isSynthesizing = false;
        }
    }
   
    private string CleanTextForSpeech(string text)
    {
        // Remove or replace common markdown and symbols that shouldn't be read aloud
        string cleaned = text;
       
        // Replace asterisks used for emphasis in markdown
        cleaned = System.Text.RegularExpressions.Regex.Replace(cleaned, @"\*\*(.*?)\*\*", "$1");
        cleaned = System.Text.RegularExpressions.Regex.Replace(cleaned, @"\*(.*?)\*", "$1");
       
        // Replace other common symbols
        cleaned = cleaned.Replace("#", "");
        cleaned = cleaned.Replace("###", "");
        cleaned = cleaned.Replace("##", "");
        cleaned = cleaned.Replace("`", "");
        cleaned = cleaned.Replace("```", "");
        cleaned = cleaned.Replace("_", " ");
       
        // Replace common markdown links with just their text
        cleaned = System.Text.RegularExpressions.Regex.Replace(cleaned, @"\[(.*?)\]\(.*?\)", "$1");
       
        // Clean up extra whitespace
        cleaned = System.Text.RegularExpressions.Regex.Replace(cleaned, @"\s+", " ").Trim();
       
        return cleaned;
    }

    private void UpdateLog(string message)
    {
        Debug.Log(message);
       
        if (logText != null)
        {
            logText.text = message;
        }
    }

    private async void OnApplicationQuit()
    {
        isQuitting = true;
        await CleanupSpeechResources();
    }

    private async void OnDestroy()
    {
        if (!isQuitting) // Only cleanup if not already cleaning up from OnApplicationQuit
        {
            await CleanupSpeechResources();
        }
    }

    private async System.Threading.Tasks.Task CleanupSpeechResources()
    {
        if (Microphone.IsRecording(deviceName))
        {
            Microphone.End(deviceName);
        }

        // Wait for any ongoing synthesis to complete
        while (isSynthesizing)
        {
            await System.Threading.Tasks.Task.Delay(100);
        }

        if (synthesizer != null)
        {
            synthesizer.Dispose();
            synthesizer = null;
        }
    }

    // Response class for JSON deserialization
    [System.Serializable]
    private class ServerResponse
    {
        public bool success;
        public string response;
        public string error;
    }
} 