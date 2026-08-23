/**
 * FREQUENCY - API Client Module (Phase 6 Capstone with Cross-Sense Fusion & Trends)
 */

const API = {
    /**
     * Fetch health and operational status.
     */
    async getStatus() {
        const startTime = performance.now();
        try {
            const response = await fetch('/api/status', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            const latencyMs = Math.round(performance.now() - startTime);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return { data, latencyMs, error: null };
        } catch (err) {
            const latencyMs = Math.round(performance.now() - startTime);
            return {
                data: null,
                latencyMs,
                error: err.message || 'Unable to connect to Flask backend'
            };
        }
    },

    /**
     * Get active daily context (Phase 4).
     */
    async getContext() {
        try {
            const response = await fetch('/api/context', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to get context:", err);
            return null;
        }
    },

    /**
     * Save daily lifestyle context (Phase 4).
     */
    async saveContext(contextData) {
        try {
            const response = await fetch('/api/context', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(contextData)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to save context:", err);
            return { status: "error", error: err.message };
        }
    },

    /**
     * Retrieve personal baseline profile and deviations (Phase 5).
     */
    async getBaseline() {
        try {
            const response = await fetch('/api/baseline', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to get baseline:", err);
            return null;
        }
    },

    /**
     * Retrieve 7-day multi-modal trends for dashboard (Phase 6).
     */
    async getTrends() {
        try {
            const response = await fetch('/api/trends', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to get trends:", err);
            return null;
        }
    },

    /**
     * Execute Cross-Sense Fusion on an evidence package (Phase 6).
     */
    async synthesizeFusion(packageData) {
        try {
            const response = await fetch('/api/fusion/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(packageData)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to synthesize fusion:", err);
            return null;
        }
    },

    /**
     * Standalone text signal analysis endpoint (Phase 2).
     */
    async analyzeText(text) {
        try {
            const response = await fetch('/api/analyze/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Text analysis error:", err);
            return { status: "error", error: err.message };
        }
    },

    /**
     * Voice audio analysis and STT endpoint (Phase 3).
     */
    async uploadAudio(audioBlob, filename = 'voice_recording.wav', clientTranscript = '') {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, filename);
            if (clientTranscript) {
                formData.append('transcript', clientTranscript);
            }

            const response = await fetch('/api/analyze/voice', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (err) {
            console.error("Voice upload error:", err);
            return { status: "error", error: err.message };
        }
    },

    /**
     * Fetch JSON schema definition.
     */
    async getSchema() {
        try {
            const response = await fetch('/api/schema', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to fetch schema:", err);
            return null;
        }
    },

    /**
     * Preview compiled evidence prompt from PromptBuilder.
     */
    async previewPrompt(packageData) {
        try {
            const response = await fetch('/api/prompt/preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(packageData)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to preview prompt:", err);
            return null;
        }
    },

    /**
     * Validate and assemble structured FrequencyInput package.
     */
    async validatePackage(packageData) {
        try {
            const response = await fetch('/api/analyze/package', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(packageData)
            });
            return await response.json();
        } catch (err) {
            return { status: 'error', error: err.message };
        }
    },

    /**
     * Retrieve list of installed models.
     */
    async getModels() {
        try {
            const response = await fetch('/api/models', {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error("Failed to fetch models:", err);
            return { models: [], count: 0 };
        }
    },

    /**
     * Stream structured FrequencyInput or prompt in real-time.
     */
    async streamMessage(payloadData, model, onChunk, onComplete, onError) {
        const startTime = performance.now();
        try {
            const bodyPayload = typeof payloadData === 'string' 
                ? { message: payloadData, model }
                : { ...payloadData, model };

            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(bodyPayload)
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                onError(errData.error || `Server returned HTTP ${response.status}`);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    
                    try {
                        const payload = JSON.parse(trimmed.slice(6));
                        if (payload.error) {
                            onError(payload.error);
                            return;
                        }
                        if (payload.chunk) {
                            onChunk(payload.chunk);
                        }
                        if (payload.done) {
                            const clientLatency = Math.round(performance.now() - startTime);
                            onComplete({
                                model: payload.model,
                                duration_ms: payload.duration_ms || clientLatency,
                                eval_count: payload.eval_count || 0
                            });
                            return;
                        }
                    } catch (parseErr) {
                        console.warn("SSE parse error:", parseErr, trimmed);
                    }
                }
            }

            onComplete({
                model: model,
                duration_ms: Math.round(performance.now() - startTime)
            });

        } catch (err) {
            onError(err.message || 'Stream connection failed');
        }
    }
};

window.API = API;
