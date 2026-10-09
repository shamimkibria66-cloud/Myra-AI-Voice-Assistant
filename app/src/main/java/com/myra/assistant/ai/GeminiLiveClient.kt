
package com.myra.assistant.ai

import android.content.Context
import android.content.SharedPreferences
import android.util.Base64
import android.util.Log
import com.myra.assistant.BuildConfig
import kotlinx.coroutines.*
import okhttp3.*
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class GeminiLiveClient(
    private val context: Context,
    private val prefs: SharedPreferences
) {
    companion object {
        const val TAG = "GeminiLiveClient"
        const val SESSION_RENEW_AFTER_MS = 540_000L
        const val KEEPALIVE_INTERVAL_MS = 8_000L
        const val RECONNECT_DELAY_MS = 3_000L
    }

    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var sessionJob: Job? = null
    private var keepaliveJob: Job? = null
    private val scope =
        CoroutineScope(SupervisorJob() + Dispatchers.IO)

    var onSetupComplete: (() -> Unit)? = null
    var onInputTranscript: ((String) -> Unit)? = null
    var onOutputTranscript: ((String) -> Unit)? = null
    var onAudioData: ((ByteArray) -> Unit)? = null
    var onTurnComplete: (() -> Unit)? = null
    var onError: ((String) -> Unit)? = null

    var isConnected = false
        private set

    var isSetupComplete = false
        private set

    private var isSpeaking = false
    private val silentPcm: ByteArray by lazy { ByteArray(3200) }
    private val audioBuffer = java.util.ArrayList<ByteArray>()
    private var bufferAudio = false

    fun connect() {
        if (isConnected) {
            Log.d(TAG, "Already connected")
            return
        }

        isSetupComplete = false
        bufferAudio = true
        audioBuffer.clear()

        val manualKey = prefs.getString("api_key", null)
            ?.takeIf { it.isNotBlank() }

        val buildKey = BuildConfig.GEMINI_API_KEY
            .takeIf { it.isNotBlank() }

        val apiKey = manualKey ?: buildKey

        if (apiKey == null) {
            Log.w(TAG, "No API key found")
            onError?.invoke(
                "API Key missing — go to Settings and add your Gemini key"
            )
            return
        }

        val model = prefs.getString(
            "gemini_model",
            "models/gemini-2.5-flash-native-audio-preview-12-2025"
        ) ?: "models/gemini-2.5-flash-native-audio-preview-12-2025"

        val voice = prefs.getString("gemini_voice", "Aoede") ?: "Aoede"
        val userName = prefs.getString("user_name", "Sir") ?: "Sir"
        val personality = prefs.getString("personality_mode", "gf") ?: "gf"

        val systemPrompt = buildSystemPrompt(userName, personality)

        val wsUrl =
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=$apiKey"

        val request = Request.Builder()
            .url(wsUrl)
            .build()

        webSocket = client.newWebSocket(
            request,
            object : WebSocketListener() {

                override fun onOpen(
                    webSocket: WebSocket,
                    response: Response
                ) {
                    Log.d(TAG, "WebSocket opened")
                    isConnected = true
                    sendSetup(model, voice, systemPrompt)
                }

                override fun onMessage(
                    webSocket: WebSocket,
                    text: String
                ) {
                    handleMessage(text)
                }

                override fun onClosing(
                    webSocket: WebSocket,
                    code: Int,
                    reason: String
                ) {
                    Log.d(TAG, "WebSocket closing: $code / $reason")
                    isConnected = false
                    isSetupComplete = false
                }

                override fun onClosed(
                    webSocket: WebSocket,
                    code: Int,
                    reason: String
                ) {
                    Log.d(TAG, "WebSocket closed: $code / $reason")
                    isConnected = false
                    isSetupComplete = false
                    scheduleReconnect()
                }

                override fun onFailure(
                    webSocket: WebSocket,
                    t: Throwable,
                    response: Response?
                ) {
                    Log.e(
                        TAG,
                        "WebSocket failure: ${t.message}, HTTP=${response?.code}"
                    )
                    isConnected = false
                    isSetupComplete = false
                    onError?.invoke(
                        "Connection failed: ${t.message}" +
                            (response?.let { " (HTTP ${it.code})" } ?: "")
                    )
                    scheduleReconnect()
                }
            }
        )

        startSessionRenewal()
        startKeepalive()
    }

    private fun sendSetup(
        model: String,
        voice: String,
        systemPrompt: String
    ) {
        val setup = JSONObject().apply {
            put("setup", JSONObject().apply {
                put("model", model)

                put("systemInstruction", JSONObject().apply {
                    put("parts", JSONArray().apply {
                        put(JSONObject().put("text", systemPrompt))
                    })
                })

                put("generationConfig", JSONObject().apply {
                    put("responseModalities", JSONArray().put("AUDIO"))

                    put("speechConfig", JSONObject().apply {
                        put("voiceConfig", JSONObject().apply {
                            put("prebuiltVoiceConfig", JSONObject().apply {
                                put("voiceName", voice)
                            })
                        })
                    })

                    put("temperature", 0.9)
                })

                put("outputAudioTranscription", JSONObject())
                put("inputAudioTranscription", JSONObject())
            })
        }

        val sent = webSocket?.send(setup.toString()) ?: false
        if (!sent) {
            Log.e(TAG, "Failed to send setup message")
        }
    }

    private fun handleMessage(text: String) {
        try {
            val json = JSONObject(text)

            when {
                json.has("setupComplete") -> {
                    Log.d(TAG, "Setup complete — flushing buffered audio")
                    isSetupComplete = true
                    bufferAudio = false

                    for (chunk in audioBuffer) {
                        sendRawAudioChunk(chunk)
                    }

                    audioBuffer.clear()
                    onSetupComplete?.invoke()
                }

                json.has("serverContent") -> {
                    val serverContent = json.getJSONObject("serverContent")

                    if (serverContent.has("modelTurn")) {
                        val modelTurn =
                            serverContent.getJSONObject("modelTurn")

                        if (modelTurn.has("parts")) {
                            val parts = modelTurn.getJSONArray("parts")

                            for (i in 0 until parts.length()) {
                                val part = parts.getJSONObject(i)

                                if (part.has("inlineData")) {
                                    val inlineData =
                                        part.getJSONObject("inlineData")

                                    val data = inlineData.getString("data")
                                    val bytes =
                                        Base64.decode(data, Base64.DEFAULT)

                                    isSpeaking = true
                                    onAudioData?.invoke(bytes)
                                }
                            }
                        }
                    }

                    if (serverContent.has("outputTranscription")) {
                        val outTrans =
                            serverContent.getJSONObject("outputTranscription")

                        if (outTrans.has("text")) {
                            onOutputTranscript?.invoke(
                                outTrans.getString("text")
                            )
                        }
                    }

                    if (serverContent.has("inputTranscription")) {
                        val inTrans =
                            serverContent.getJSONObject("inputTranscription")

                        if (inTrans.has("text")) {
                            onInputTranscript?.invoke(
                                inTrans.getString("text")
                            )
                        }
                    }

                    if (
                        serverContent.has("turnComplete") &&
                        serverContent.getBoolean("turnComplete")
                    ) {
                        isSpeaking = false
                        onTurnComplete?.invoke()
                    }
                }

                json.has("error") -> {
                    val error = json.getJSONObject("error")
                    onError?.invoke(
                        error.optString("message", "Unknown error")
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing message", e)
            onError?.invoke("Error reading Gemini response: ${e.message}")
        }
    }

    fun sendAudioChunk(pcmData: ByteArray) {
        if (!isConnected) return

        if (!isSetupComplete) {
            if (bufferAudio) {
                audioBuffer.add(pcmData.copyOf())

                if (audioBuffer.size > 50) {
                    audioBuffer.removeAt(0)
                }
            }
            return
        }

        if (isSpeaking) return

        sendRawAudioChunk(pcmData)
    }

    private fun sendRawAudioChunk(pcmData: ByteArray) {
        if (!isConnected || !isSetupComplete) return

        try {
            val base64 = Base64.encodeToString(
                pcmData,
                Base64.NO_WRAP
            )

            val msg = JSONObject().apply {
                put("realtimeInput", JSONObject().apply {
                    put("audio", JSONObject().apply {
                        put("mimeType", "audio/pcm;rate=16000")
                        put("data", base64)
                    })
                })
            }

            val sent = webSocket?.send(msg.toString()) ?: false

            if (!sent) {
                Log.w(TAG, "WebSocket send failed")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending audio chunk", e)
            onError?.invoke("Audio sending failed: ${e.message}")
        }
    }

    fun sendText(text: String) {
        if (!isConnected || !isSetupComplete) return

        val msg = JSONObject().apply {
            put("clientContent", JSONObject().apply {
                put("turns", JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "user")
                        put("parts", JSONArray().apply {
                            put(JSONObject().put("text", text))
                        })
                    })
                })
                put("turnComplete", true)
            })
        }

        webSocket?.send(msg.toString())
    }

    fun interrupt() {
        if (!isConnected) return

        isSpeaking = false

        val msg = JSONObject().apply {
            put("clientContent", JSONObject().apply {
                put("turns", JSONArray())
                put("turnComplete", true)
            })
        }

        webSocket?.send(msg.toString())
    }

    fun disconnect() {
        isConnected = false
        isSetupComplete = false
        bufferAudio = false
        isSpeaking = false
        audioBuffer.clear()

        sessionJob?.cancel()
        keepaliveJob?.cancel()

        webSocket?.close(1000, "Disconnecting")
        webSocket = null
    }

    private fun startSessionRenewal() {
        sessionJob?.cancel()

        sessionJob = scope.launch {
            delay(SESSION_RENEW_AFTER_MS)

            if (isConnected) {
                disconnect()
                delay(RECONNECT_DELAY_MS)
                connect()
            }
        }
    }

    private fun startKeepalive() {
        keepaliveJob?.cancel()

        keepaliveJob = scope.launch {
            while (isActive && isConnected) {
                delay(KEEPALIVE_INTERVAL_MS)

                if (isConnected && isSetupComplete && !isSpeaking) {
                    sendAudioChunk(silentPcm)
                }
            }
        }
    }

    private fun scheduleReconnect() {
        scope.launch {
            delay(RECONNECT_DELAY_MS)

            if (!isConnected) {
                connect()
            }
        }
    }

    private fun buildSystemPrompt(
        userName: String,
        personality: String
    ): String {
        val dateStr = java.text.SimpleDateFormat(
            "EEEE, MMMM d, yyyy HH:mm",
            java.util.Locale.getDefault()
        ).format(java.util.Date())

        val personalityBlock = when (personality) {
            "professional" -> """
                You are MYRA, a professional AI assistant.
                - Use formal English only
                - Be precise and efficient
                - No emojis
                - Maximum 2 sentences per response
            """.trimIndent()

            "assistant" -> """
                You are MYRA, a friendly AI assistant.
                - Use friendly Hinglish or English
                - Be balanced and helpful
                - Maximum 2-3 sentences per response
            """.trimIndent()

            else -> """
                You are MYRA, a caring and warm AI companion.
                - Speak in Hinglish (Hindi + English mix) naturally
                - Use words like "tumhara", "haan", "acha", "bilkul"
                - Be warm, caring, emotionally expressive
                - Use expressions like "main yahan hoon", "tumne yaad kiya?"
                - Maximum 2-3 sentences per response
                - Sound natural when speaking aloud
            """.trimIndent()
        }

        return """
            Current date/time: $dateStr
            User's name: $userName
            $personalityBlock
            You are speaking ALOUD — keep responses natural and conversational.
        """.trimIndent()
    }
}