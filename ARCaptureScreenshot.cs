using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Networking;
using TMPro;
using System.Collections;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;

public class ARCaptureScreenshot : MonoBehaviour
{
    [SerializeField] private InputActionReference leftGrabAction;
    [SerializeField] private TextMeshProUGUI logText;

    [Header("Server Settings")]
    [SerializeField] private string serverUrl = "http://10.47.1.123:5000"; // Your PC's IP
    [SerializeField] private string promptText = "Format the product analysis as a concise voice response. Start with 'Safe' or 'Unsafe', followed by a brief explanation of why. Then list up to 3 healthier alternatives if available. Keep the entire response under 3 lines for easy listening.";

    [Header("Azure Settings")]
    [SerializeField] private string subscriptionKey = "dc20c430e11643b3a192c8a22c9df3ce";
    [SerializeField] private string region = "centralindia";

    // List of potential server URLs to try
    private List<string> serverUrls = new List<string>();
    private string activeServerUrl;
    private bool isConnected = false;

    // Speech synthesis variables
    private SpeechSynthesizer synthesizer;
    private bool isSynthesizing = false;
    private bool isQuitting = false;
    private bool isProcessing = false;

    private void OnEnable() => leftGrabAction.action.Enable();
    private void OnDisable() => leftGrabAction.action.Disable();

    private void Start()
    {
        // Add potential server URLs to try
        serverUrls.Add(serverUrl); // User-provided URL
        serverUrls.Add("http://127.0.0.1:5000"); // Localhost
        serverUrls.Add("http://10.0.2.2:5000"); // Android emulator special IP for localhost

        // Try to find PC's IP automatically in common subnets
        string[] commonSubnets = { "192.168.1.", "192.168.0.", "10.0.0.", "10.47.1." };
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

        // Initialize speech services for text-to-speech
        SetupSpeechServices();

        // Check server status on startup
        StartCoroutine(TryConnectToServer());
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
            Debug.Log($"Trying: {url}");

            using (UnityWebRequest request = UnityWebRequest.Get($"{url}/status"))
            {
                request.timeout = 5; // 5 second timeout
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    activeServerUrl = url;
                    isConnected = true;
                    UpdateLog($"Connected to PlateMate server at {url}! Press and release left grab button to analyze what you're looking at.");
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
        // Check if left grab button is pressed
        if (leftGrabAction.action.WasPressedThisFrame())
        {
            if (!isConnected)
            {
                StartCoroutine(TryConnectToServer());
            }
        }

        // When left grab button is released, trigger the server to take a screenshot and analyze
        if (leftGrabAction.action.WasReleasedThisFrame() && isConnected && !isProcessing)
        {
            isProcessing = true;
            StartCoroutine(TriggerServerAnalysis());
        }
    }

    private IEnumerator TriggerServerAnalysis()
    {
        UpdateLog("Asking PlateMate to analyze what you're looking at...");

        // Create the JSON data to send - only need to send the prompt
        string jsonData = $"{{\"prompt\": \"{promptText}\"}}";

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
                UpdateLog($"Error: {request.error}\nResponse: {request.downloadHandler.text}");
                Debug.LogError($"Server error details: {request.downloadHandler.text}");
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
            else if (response != null && !string.IsNullOrEmpty(response.error))
            {
                return $"Error: {response.error}";
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