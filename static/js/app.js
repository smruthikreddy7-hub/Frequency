/**
 * FREQUENCY - Multimodal Conversational Controller with Cross-Sense Fusion Matrix & PCM WAV Recording
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const welcomeHero = document.getElementById('welcome-hero');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const activeModelBadge = document.getElementById('active-model-badge');
    const backendLatency = document.getElementById('backend-latency');
    const modelSelect = document.getElementById('model-select');
    const newChatBtn = document.getElementById('new-chat-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const toggleSchemaBtn = document.getElementById('toggle-schema-btn');
    const closeSchemaBtn = document.getElementById('close-schema-btn');
    const schemaDrawer = document.getElementById('schema-drawer');
    const schemaJsonView = document.getElementById('schema-json-view');
    const compiledPromptView = document.getElementById('compiled-prompt-view');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');

    // Context Bar UI (Phase 4)
    const toggleContextBtn = document.getElementById('toggle-context-btn');
    const headerContextSummary = document.getElementById('header-context-summary');
    const contextBar = document.getElementById('context-bar');
    const ctxSleepInput = document.getElementById('ctx-sleep-input');
    const ctxWorkloadSelect = document.getElementById('ctx-workload-select');
    const ctxMoodInput = document.getElementById('ctx-mood-input');
    const ctxMoodVal = document.getElementById('ctx-mood-val');
    const saveContextBtn = document.getElementById('save-context-btn');

    // Trends Dashboard Modal (Phase 6)
    const toggleTrendsBtn = document.getElementById('toggle-trends-btn');
    const closeTrendsBtn = document.getElementById('close-trends-btn');
    const trendsModal = document.getElementById('trends-modal');
    const trendsChartContainer = document.getElementById('trends-chart-container');

    // Fusion Matrix UI (Phase 6 Capstone)
    const fusionEnergyVal = document.getElementById('fusion-energy-val');
    const fusionEnergyBar = document.getElementById('fusion-energy-bar');
    const fusionEnergyBadge = document.getElementById('fusion-energy-badge');
    const fusionFocusVal = document.getElementById('fusion-focus-val');
    const fusionFocusBar = document.getElementById('fusion-focus-bar');
    const fusionAlignVal = document.getElementById('fusion-align-val');
    const fusionArchetypeTitle = document.getElementById('fusion-archetype-title');
    const fusionArchetypeDesc = document.getElementById('fusion-archetype-desc');
    const fusionRecsContainer = document.getElementById('fusion-recs-container');

    // Baseline UI (Phase 5)
    const baseSleepVal = document.getElementById('base-sleep-val');
    const baseEnergyVal = document.getElementById('base-energy-val');
    const baseTempoVal = document.getElementById('base-tempo-val');
    const baseDeviationsContainer = document.getElementById('base-deviations-container');
    const baseCorrelationsContainer = document.getElementById('base-correlations-container');

    // Voice Controls (Phase 3)
    const micRecordBtn = document.getElementById('mic-record-btn');
    const recordingTimer = document.getElementById('recording-time');
    const uploadAudioBtn = document.getElementById('upload-audio-btn');
    const audioFileInput = document.getElementById('audio-file-input');

    // Voice Acoustics UI Elements (Phase 3)
    const sigPitchHz = document.getElementById('sig-pitch-hz');
    const sigPitchVariation = document.getElementById('sig-pitch-variation');
    const sigVocalEnergy = document.getElementById('sig-vocal-energy');
    const sigEnergyBadge = document.getElementById('sig-energy-badge');
    const sigSpeechRate = document.getElementById('sig-speech-rate');
    const sigPauseRatio = document.getElementById('sig-pause-ratio');
    const sigAudioDuration = document.getElementById('sig-audio-duration');
    const sigTranscriptPreview = document.getElementById('sig-transcript-preview');

    // Text Signals UI Elements (Phase 2)
    const sigSentimentBadge = document.getElementById('sig-sentiment-badge');
    const sigValenceScore = document.getElementById('sig-valence-score');
    const sigValenceFill = document.getElementById('sig-valence-fill');
    const sigEmotionsContainer = document.getElementById('sig-emotions-container');
    const sigThemesContainer = document.getElementById('sig-themes-container');
    const sigCognitiveLoad = document.getElementById('sig-cognitive-load');
    const sigWordCount = document.getElementById('sig-word-count');
    const sigConfidence = document.getElementById('sig-confidence');

    // Drawer Tabs
    const tabBtnFusion = document.getElementById('tab-btn-fusion');
    const tabBtnBaseline = document.getElementById('tab-btn-baseline');
    const tabBtnVoice = document.getElementById('tab-btn-voice');
    const tabBtnText = document.getElementById('tab-btn-text');
    const tabBtnPrompt = document.getElementById('tab-btn-prompt');
    const tabBtnJson = document.getElementById('tab-btn-json');
    const drawerViewFusion = document.getElementById('drawer-view-fusion');
    const drawerViewBaseline = document.getElementById('drawer-view-baseline');
    const drawerViewVoice = document.getElementById('drawer-view-voice');
    const drawerViewText = document.getElementById('drawer-view-text');
    const drawerViewPrompt = document.getElementById('drawer-view-prompt');
    const drawerViewJson = document.getElementById('drawer-view-json');

    let currentModel = modelSelect ? modelSelect.value : 'qwen3:8b';
    let isProcessing = false;
    let currentEvidencePackage = null;
    let activeContext = { sleep_hours: 7.0, workload: 'normal', mood_score: 7, sleep_quality: 'moderate' };
    let activeBaseline = null;
    let activeFusion = null;

    // Audio Context & PCM WAV Recording
    let audioContext = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let audioInput = null;
    let pcmBuffers = [];
    let isRecording = false;
    let recordingInterval = null;
    let recordingSeconds = 0;
    let speechRecognitionInstance = null;
    let liveVoiceTranscript = '';

    /**
     * Format time for message timestamp.
     */
    function formatTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    /**
     * Escape HTML helper.
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.innerText = text;
        return div.innerHTML;
    }

    /**
     * Markdown formatter for assistant bubbles.
     */
    function formatMarkdown(text) {
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/### (.*?)\n/g, '<h4 style="color:var(--accent-emerald);margin-top:10px;margin-bottom:4px;font-size:13px;font-weight:700;">$1</h4>');
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre class="code-block mono">$1</pre>');
        formatted = formatted.replace(/`([^`]+)`/g, '<code class="mono" style="background:#111520;padding:2px 6px;border-radius:4px;border:1px solid #232938;">$1</code>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/\n/g, '<br>');
        return formatted;
    }

    /**
     * Encode float audio samples into a standard 16-bit PCM WAV Blob.
     */
    function encodeWAV(samples, sampleRate = 16000) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        /* RIFF identifier */
        writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + samples.length * 2, true);
        /* RIFF type */
        writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw PCM) */
        view.setUint16(20, 1, true);
        /* channel count (mono) */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sample rate * block align) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (channel count * bytes per sample) */
        view.setUint16(32, 2, true);
        /* bits per sample */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, samples.length * 2, true);

        // Float to 16-bit PCM conversion
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /**
     * Construct a standardized FrequencyInput structure.
     */
    function buildFrequencyInput(promptText, voiceData = null, textData = null) {
        return {
            user_prompt: promptText,
            text_data: textData,
            voice_data: voiceData,
            contextual_data: activeContext,
            baseline_data: activeBaseline || {},
            fusion_data: activeFusion || null,
            metadata: {
                timestamp: new Date().toISOString(),
                session_id: 'session_' + Math.random().toString(36).substring(2, 9),
                schema_version: '1.4.0'
            }
        };
    }

    /**
     * Render Cross-Sense Fusion Matrix in drawer (Phase 6 Capstone).
     */
    function renderFusionTelemetry(fusionData) {
        if (!fusionData) return;

        const energy = fusionData.energy_index !== undefined ? fusionData.energy_index : 70.0;
        const focus = fusionData.focus_index !== undefined ? fusionData.focus_index : 75.0;
        const align = fusionData.alignment_score !== undefined ? fusionData.alignment_score : 85.0;

        if (fusionEnergyVal) fusionEnergyVal.textContent = `${energy}%`;
        if (fusionEnergyBar) fusionEnergyBar.style.width = `${energy}%`;
        if (fusionEnergyBadge) {
            if (energy >= 75) {
                fusionEnergyBadge.textContent = 'High Flow';
                fusionEnergyBadge.className = 'badge badge-success';
            } else if (energy >= 55) {
                fusionEnergyBadge.textContent = 'Balanced';
                fusionEnergyBadge.className = 'badge badge-info';
            } else {
                fusionEnergyBadge.textContent = 'Low Stamina';
                fusionEnergyBadge.className = 'badge badge-danger';
            }
        }

        if (fusionFocusVal) fusionFocusVal.textContent = `${focus}%`;
        if (fusionFocusBar) fusionFocusBar.style.width = `${focus}%`;
        if (fusionAlignVal) fusionAlignVal.textContent = `${align}%`;

        if (fusionArchetypeTitle) {
            fusionArchetypeTitle.textContent = fusionData.archetype || 'Steady Baseline Equilibrium';
        }
        if (fusionArchetypeDesc) {
            fusionArchetypeDesc.textContent = fusionData.archetype_description || 'Sensory streams are aligned and balanced.';
        }

        if (fusionRecsContainer) {
            const recs = fusionData.recommendations || [];
            if (recs.length === 0) {
                fusionRecsContainer.innerHTML = '<span class="tag-pill subtle">Maintain steady single-tasking flow.</span>';
            } else {
                fusionRecsContainer.innerHTML = recs.map(r => {
                    return `<span class="tag-pill correlation">💡 ${escapeHtml(r)}</span>`;
                }).join('');
            }
        }
    }

    /**
     * Render Baseline & Deviations in drawer (Phase 5).
     */
    function renderBaselineTelemetry(baselineData) {
        if (!baselineData) return;

        if (baseSleepVal) baseSleepVal.textContent = (baselineData.baseline_sleep_hours || 7.4) + 'h';
        if (baseEnergyVal) baseEnergyVal.textContent = baselineData.baseline_vocal_energy_rms || 0.038;
        if (baseTempoVal) baseTempoVal.textContent = (baselineData.baseline_speech_rate_wpm || 145.0) + ' WPM';

        const devs = baselineData.deviations || {};

        if (baseDeviationsContainer) {
            const devKeys = Object.keys(devs).filter(k => k.endsWith('_pct'));
            if (devKeys.length === 0) {
                baseDeviationsContainer.innerHTML = '<span class="tag-pill subtle">Within normal baseline ranges</span>';
            } else {
                baseDeviationsContainer.innerHTML = devKeys.map(k => {
                    const label = k.replace('_pct', '').replace('_', ' ').toUpperCase();
                    const val = devs[k];
                    const sign = val > 0 ? '+' : '';
                    const cls = val < -15 ? 'tag-pill shift-neg' : val > 15 ? 'tag-pill shift-pos' : 'tag-pill subtle';
                    return `<span class="${cls}">Δ ${label}: ${sign}${val}%</span>`;
                }).join(' ');
            }
        }

        if (baseCorrelationsContainer) {
            const correlations = devs.cross_modal_correlations || [];
            if (correlations.length === 0) {
                baseCorrelationsContainer.innerHTML = '<span class="tag-pill subtle">No adverse multi-sense friction identified today.</span>';
            } else {
                baseCorrelationsContainer.innerHTML = correlations.map(c => {
                    return `<span class="tag-pill correlation">⚡ ${escapeHtml(c)}</span>`;
                }).join('');
            }
        }
    }

    /**
     * Render Voice Acoustic Signals in drawer (Phase 3).
     */
    function renderVoiceSignalTelemetry(voiceSignals) {
        if (!voiceSignals) return;

        if (sigPitchHz) sigPitchHz.textContent = voiceSignals.mean_pitch_hz ? `${voiceSignals.mean_pitch_hz} Hz` : '-- Hz';
        if (sigPitchVariation) {
            const stdev = voiceSignals.pitch_variation_stdev ? Number(voiceSignals.pitch_variation_stdev).toFixed(1) : '0.0';
            sigPitchVariation.textContent = `±${stdev} Hz`;
        }
        if (sigVocalEnergy) {
            const rms = voiceSignals.vocal_energy_rms !== undefined ? Number(voiceSignals.vocal_energy_rms).toFixed(4) : '0.0000';
            sigVocalEnergy.textContent = rms;
        }
        if (sigEnergyBadge) {
            const rms = voiceSignals.vocal_energy_rms || 0;
            if (rms > 0.05) {
                sigEnergyBadge.textContent = 'High Energy';
                sigEnergyBadge.className = 'badge badge-success';
            } else if (rms > 0.015) {
                sigEnergyBadge.textContent = 'Moderate Energy';
                sigEnergyBadge.className = 'badge badge-info';
            } else {
                sigEnergyBadge.textContent = 'Low / Subdued';
                sigEnergyBadge.className = 'badge badge-warning';
            }
        }
        if (sigSpeechRate) sigSpeechRate.textContent = voiceSignals.speech_rate_wpm ? `${voiceSignals.speech_rate_wpm} WPM` : '-- WPM';
        if (sigPauseRatio) {
            const pct = voiceSignals.pause_duration_ratio !== undefined ? Math.round(voiceSignals.pause_duration_ratio * 100) : 0;
            sigPauseRatio.textContent = `${pct}%`;
        }
        if (sigAudioDuration) sigAudioDuration.textContent = voiceSignals.audio_duration_seconds ? `${voiceSignals.audio_duration_seconds}s` : '0.0s';
        if (sigTranscriptPreview) sigTranscriptPreview.textContent = voiceSignals.transcript || 'No speech transcribed.';
    }

    /**
     * Render Text Signals in drawer (Phase 2).
     */
    function renderTextSignalTelemetry(signals) {
        if (!signals) return;

        if (sigSentimentBadge) {
            const polarity = signals.sentiment || 'neutral';
            sigSentimentBadge.textContent = polarity.toUpperCase();
            if (polarity === 'positive') sigSentimentBadge.className = 'badge badge-success';
            else if (polarity === 'negative') sigSentimentBadge.className = 'badge badge-danger';
            else sigSentimentBadge.className = 'badge badge-neutral';
        }

        if (sigValenceScore) {
            const v = signals.emotional_valence !== undefined ? signals.emotional_valence : 0.0;
            sigValenceScore.textContent = (v > 0 ? '+' : '') + Number(v).toFixed(2);
        }

        if (sigValenceFill) {
            const v = signals.emotional_valence !== undefined ? signals.emotional_valence : 0.0;
            const pct = Math.max(5, Math.min(95, ((v + 1.0) / 2.0) * 100));
            sigValenceFill.style.width = pct + '%';
        }

        if (sigEmotionsContainer) {
            const emotions = signals.emotional_signals || {};
            const emotionKeys = Object.keys(emotions);
            if (emotionKeys.length === 0) {
                sigEmotionsContainer.innerHTML = '<span class="tag-pill subtle">Neutral affect</span>';
            } else {
                sigEmotionsContainer.innerHTML = emotionKeys.map(k => {
                    const label = k.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                    const valPct = Math.round(emotions[k] * 100);
                    return `<span class="tag-pill emotion">${label} (${valPct}%)</span>`;
                }).join('');
            }
        }

        if (sigThemesContainer) {
            const themes = signals.detected_themes || [];
            if (themes.length === 0) {
                sigThemesContainer.innerHTML = '<span class="tag-pill subtle">General reflection</span>';
            } else {
                sigThemesContainer.innerHTML = themes.map(t => {
                    const label = '#' + t.replace('_', ' ');
                    return `<span class="tag-pill theme">${label}</span>`;
                }).join('');
            }
        }

        if (sigCognitiveLoad) sigCognitiveLoad.textContent = (signals.cognitive_load || 'Moderate').toUpperCase();
        if (sigWordCount) sigWordCount.textContent = signals.word_count || 0;
        if (sigConfidence) {
            const conf = signals.confidence !== undefined ? Math.round(signals.confidence * 100) + '%' : '--';
            sigConfidence.textContent = conf;
        }
    }

    /**
     * Update Schema Inspector & Compiled Prompt views live.
     */
    async function updatePromptAndSchemaViews(pkg) {
        currentEvidencePackage = pkg;

        // Auto run fusion if missing
        if (!pkg.fusion_data) {
            const fusionRes = await window.API.synthesizeFusion(pkg);
            if (fusionRes && fusionRes.fusion) {
                pkg.fusion_data = fusionRes.fusion;
                activeFusion = fusionRes.fusion;
            }
        }

        if (pkg.fusion_data) renderFusionTelemetry(pkg.fusion_data);
        if (pkg.baseline_data) renderBaselineTelemetry(pkg.baseline_data);
        if (pkg.voice_data) renderVoiceSignalTelemetry(pkg.voice_data);
        if (pkg.text_data) renderTextSignalTelemetry(pkg.text_data);

        if (schemaJsonView) {
            schemaJsonView.textContent = JSON.stringify(pkg, null, 2);
        }

        const previewData = await window.API.previewPrompt(pkg);
        if (previewData && compiledPromptView) {
            compiledPromptView.textContent = previewData.evidence_prompt;
        }
    }

    /**
     * Append user message to thread.
     */
    function appendUserMessage(content, isVoice = false) {
        if (welcomeHero && welcomeHero.style.display !== 'none') {
            welcomeHero.style.display = 'none';
        }

        const voiceBadge = isVoice ? '<span class="badge badge-info" style="margin-right:6px;">🎙️ Spoken</span>' : '';

        const row = document.createElement('div');
        row.className = 'message-row user';
        row.innerHTML = `
            <div class="message-bubble">
                <div class="message-content">${voiceBadge}${escapeHtml(content).replace(/\n/g, '<br>')}</div>
                <div class="message-meta">${formatTime()}</div>
            </div>
            <div class="avatar user-avatar">U</div>
        `;
        chatMessages.appendChild(row);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Create an active assistant message placeholder for streaming.
     */
    function createStreamingAssistantRow() {
        if (welcomeHero && welcomeHero.style.display !== 'none') {
            welcomeHero.style.display = 'none';
        }

        const row = document.createElement('div');
        row.className = 'message-row assistant';
        row.innerHTML = `
            <div class="avatar assistant-avatar">⚡</div>
            <div class="message-bubble">
                <div class="message-content stream-content"><span class="stream-cursor"></span></div>
                <div class="message-meta stream-meta">${formatTime()} • ${currentModel} • <span class="streaming-label">Synthesizing...</span></div>
                <div class="message-telemetry-container"></div>
            </div>
        `;
        chatMessages.appendChild(row);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const contentEl = row.querySelector('.stream-content');
        const metaEl = row.querySelector('.stream-meta');
        const telemetryContainer = row.querySelector('.message-telemetry-container');
        return { row, contentEl, metaEl, telemetryContainer };
    }

    /**
     * Build interactive per-message telemetry card with Fusion, Baseline, Voice, and Text.
     */
    function attachMessageTelemetry(container, inputPkg, latencyMs) {
        if (!container || !inputPkg) return;

        const textSignals = inputPkg.text_data || {};
        const voiceSignals = inputPkg.voice_data || null;
        const baselineData = inputPkg.baseline_data || {};
        const fusionData = inputPkg.fusion_data || activeFusion || {};
        const devs = baselineData.deviations || {};

        const polarity = textSignals.sentiment || 'neutral';
        const valence = textSignals.emotional_valence !== undefined ? textSignals.emotional_valence : 0.0;
        const valStr = (valence > 0 ? '+' : '') + Number(valence).toFixed(2);
        
        const emotions = textSignals.emotional_signals || {};
        const emotionTags = Object.keys(emotions).map(k => {
            const label = k.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            return `<span class="tag-pill emotion">${label} (${Math.round(emotions[k] * 100)}%)</span>`;
        }).join(' ');

        const themes = textSignals.detected_themes || [];
        const themeTags = themes.map(t => `<span class="tag-pill theme">#${t.replace('_', ' ')}</span>`).join(' ');

        let fusionSection = '';
        if (fusionData && fusionData.energy_index !== undefined) {
            fusionSection = `
                <div class="msg-telemetry-row">
                    <span class="stat-label">Cross-Sense Fusion:</span>
                    <span class="tag-pill shift-pos">⚡ Energy: ${fusionData.energy_index}%</span>
                    <span class="tag-pill theme">Focus: ${fusionData.focus_index}%</span>
                    <span class="tag-pill correlation">${escapeHtml(fusionData.archetype || 'Balanced')}</span>
                </div>
            `;
        }

        let voiceSection = '';
        if (voiceSignals) {
            const pitch = voiceSignals.mean_pitch_hz ? `${voiceSignals.mean_pitch_hz} Hz` : '--';
            const rms = voiceSignals.vocal_energy_rms !== undefined ? Number(voiceSignals.vocal_energy_rms).toFixed(4) : '--';
            const wpm = voiceSignals.speech_rate_wpm ? `${voiceSignals.speech_rate_wpm} WPM` : '--';
            const pause = voiceSignals.pause_duration_ratio !== undefined ? `${Math.round(voiceSignals.pause_duration_ratio * 100)}%` : '--';

            voiceSection = `
                <div class="msg-telemetry-row" style="border-top:1px dashed var(--border-subtle);padding-top:6px;margin-top:4px;">
                    <span class="stat-label">Voice Acoustics:</span>
                    <span class="tag-pill voice">Pitch: ${pitch}</span>
                    <span class="tag-pill voice">Energy: ${rms} RMS</span>
                    <span class="tag-pill voice">Tempo: ${wpm}</span>
                    <span class="tag-pill voice">Pauses: ${pause}</span>
                </div>
            `;
        }

        let baselineSection = '';
        const devKeys = Object.keys(devs).filter(k => k.endsWith('_pct'));
        if (devKeys.length > 0) {
            const devPills = devKeys.map(k => {
                const label = k.replace('_pct', '').replace('_', ' ').toUpperCase();
                const val = devs[k];
                const sign = val > 0 ? '+' : '';
                const cls = val < -15 ? 'tag-pill shift-neg' : val > 15 ? 'tag-pill shift-pos' : 'tag-pill subtle';
                return `<span class="${cls}">Δ ${label}: ${sign}${val}%</span>`;
            }).join(' ');

            baselineSection = `
                <div class="msg-telemetry-row" style="border-top:1px dashed var(--border-subtle);padding-top:6px;margin-top:4px;">
                    <span class="stat-label">Baseline Shifts:</span>
                    ${devPills}
                </div>
            `;
        }

        const uniqueId = 'telem_' + Math.random().toString(36).substring(2, 7);

        container.innerHTML = `
            <div style="margin-top: 6px;">
                <button type="button" class="msg-telemetry-toggle" data-target="${uniqueId}">
                    📊 Signal Telemetry
                </button>
                <div id="${uniqueId}" class="msg-telemetry-card" style="display: none;">
                    <div class="msg-telemetry-header">
                        <span>Cross-Sense Fusion & Multimodal Telemetry</span>
                        <span class="mono">${latencyMs}ms</span>
                    </div>
                    ${fusionSection}
                    <div class="msg-telemetry-row">
                        <span class="stat-label">Sentiment:</span>
                        <span class="badge ${polarity === 'positive' ? 'badge-success' : polarity === 'negative' ? 'badge-danger' : 'badge-neutral'}">${polarity.toUpperCase()} (${valStr})</span>
                        <span class="stat-label" style="margin-left:8px;">Load:</span>
                        <span class="mono">${(textSignals.cognitive_load || 'Moderate').toUpperCase()}</span>
                    </div>
                    ${emotionTags ? `<div class="msg-telemetry-row"><span class="stat-label">Emotions:</span> ${emotionTags}</div>` : ''}
                    ${themeTags ? `<div class="msg-telemetry-row"><span class="stat-label">Themes:</span> ${themeTags}</div>` : ''}
                    ${voiceSection}
                    ${baselineSection}
                </div>
            </div>
        `;

        const btn = container.querySelector(`[data-target="${uniqueId}"]`);
        const card = container.querySelector(`#${uniqueId}`);
        if (btn && card) {
            btn.addEventListener('click', () => {
                const isHidden = card.style.display === 'none';
                card.style.display = isHidden ? 'flex' : 'none';
                btn.style.color = isHidden ? 'var(--accent-emerald)' : 'var(--text-secondary)';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
        }
    }

    /**
     * Set busy state.
     */
    function setBusy(busy) {
        isProcessing = busy;
        if (sendBtn) sendBtn.disabled = busy;
        if (userInput) userInput.disabled = busy;
        if (micRecordBtn) micRecordBtn.disabled = busy;
        if (uploadAudioBtn) uploadAudioBtn.disabled = busy;
        if (!busy && userInput) {
            userInput.focus();
        }
    }

    /**
     * Dispatch reasoning request to backend with full multimodal package.
     */
    async function dispatchReasoningPackage(inputPackage, isVoice = false) {
        setBusy(true);

        updatePromptAndSchemaViews(inputPackage);
        appendUserMessage(inputPackage.user_prompt, isVoice);

        const { contentEl, metaEl, telemetryContainer } = createStreamingAssistantRow();
        let accumulatedText = '';

        await window.API.streamMessage(
            inputPackage,
            currentModel,
            // onChunk
            (chunk) => {
                accumulatedText += chunk;
                contentEl.innerHTML = formatMarkdown(accumulatedText) + '<span class="stream-cursor"></span>';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            },
            // onComplete
            (meta) => {
                setBusy(false);
                contentEl.innerHTML = formatMarkdown(accumulatedText);
                const latency = meta.duration_ms || 0;
                metaEl.innerHTML = `${formatTime()} • ${meta.model || currentModel} • ${latency}ms`;
                if (backendLatency) {
                    backendLatency.textContent = `Latency: ${latency} ms`;
                }

                attachMessageTelemetry(telemetryContainer, inputPackage, latency);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            },
            // onError
            (errorMsg) => {
                setBusy(false);
                contentEl.innerHTML = `<span style="color:var(--accent-rose);">⚠️ Error: ${escapeHtml(errorMsg)}</span>`;
                metaEl.innerHTML = `${formatTime()} • Error`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        );
    }

    /**
     * Handle sending a typed text message.
     */
    async function handleSendMessage(messageText) {
        const text = (messageText || (userInput ? userInput.value : '')).trim();
        if (!text || isProcessing) return;

        if (userInput) {
            userInput.value = '';
            userInput.style.height = 'auto';
        }

        const inputPackage = buildFrequencyInput(text);
        
        // Extract text signals
        const analysisRes = await window.API.analyzeText(text);
        if (analysisRes && analysisRes.signals) {
            inputPackage.text_data = analysisRes.signals;
        }

        await dispatchReasoningPackage(inputPackage, false);
    }

    /**
     * Process an audio recording blob or uploaded file (Phase 3).
     */
    async function processAudioIntake(audioBlob, filename = 'voice_recording.wav', clientTranscript = '') {
        if (userInput) {
            userInput.placeholder = 'Analyzing audio acoustics and transcribing speech...';
        }
        setBusy(true);

        const voiceRes = await window.API.uploadAudio(audioBlob, filename, clientTranscript);
        
        if (userInput) {
            userInput.placeholder = 'Share your thoughts, speak via mic 🎙️, or upload an MP3/WAV...';
        }

        if (voiceRes.status !== 'success') {
            alert(`Voice Analysis Error: ${voiceRes.error || 'Failed to process audio'}`);
            setBusy(false);
            return;
        }

        // Use transcribed words or client transcript
        const transcript = voiceRes.transcript || clientTranscript || 'Voice note shared.';
        const voiceSignals = voiceRes.voice_signals;
        const textSignals = voiceRes.text_signals;

        const inputPackage = buildFrequencyInput(transcript, voiceSignals, textSignals);

        await dispatchReasoningPackage(inputPackage, true);
    }

    /**
     * Start high-fidelity PCM WAV recording using AudioContext.
     */
    async function startRecording() {
        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioContextClass({ sampleRate: 16000 });
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            pcmBuffers = [];
            liveVoiceTranscript = '';

            // Setup speech recognition for live visual preview in browser
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRec) {
                speechRecognitionInstance = new SpeechRec();
                speechRecognitionInstance.continuous = true;
                speechRecognitionInstance.interimResults = true;
                speechRecognitionInstance.onresult = (event) => {
                    let interim = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        interim += event.results[i][0].transcript;
                    }
                    liveVoiceTranscript = interim;
                    if (userInput) {
                        userInput.value = liveVoiceTranscript;
                    }
                };
                try { speechRecognitionInstance.start(); } catch(e){}
            }

            audioInput = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                pcmBuffers.push(new Float32Array(channelData));
            };

            audioInput.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            isRecording = true;

            if (micRecordBtn) {
                micRecordBtn.classList.add('recording');
                micRecordBtn.title = 'Recording... Click to Stop & Send';
            }
            if (recordingTimer) {
                recordingTimer.style.display = 'inline';
                recordingSeconds = 0;
                recordingTimer.textContent = '0:00';
                recordingInterval = setInterval(() => {
                    recordingSeconds++;
                    const mins = Math.floor(recordingSeconds / 60);
                    const secs = recordingSeconds % 60;
                    recordingTimer.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
                }, 1000);
            }

        } catch (err) {
            console.error('Microphone access error:', err);
            alert(`Microphone access error: ${err.message}`);
        }
    }

    /**
     * Stop recording, encode PCM to WAV, and submit.
     */
    async function stopRecording() {
        if (!isRecording) return;
        isRecording = false;

        if (recordingInterval) {
            clearInterval(recordingInterval);
            recordingInterval = null;
        }
        if (micRecordBtn) {
            micRecordBtn.classList.remove('recording');
            micRecordBtn.title = 'Speak to FREQUENCY (Click to Record / Stop)';
        }
        if (recordingTimer) {
            recordingTimer.style.display = 'none';
        }

        if (speechRecognitionInstance) {
            try { speechRecognitionInstance.stop(); } catch(e){}
        }

        if (scriptProcessor && audioInput) {
            scriptProcessor.disconnect();
            audioInput.disconnect();
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }

        const sampleRate = audioContext ? audioContext.sampleRate : 16000;
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        // Merge float buffers
        let totalSamples = 0;
        for (let i = 0; i < pcmBuffers.length; i++) {
            totalSamples += pcmBuffers[i].length;
        }

        const merged = new Float32Array(totalSamples);
        let offset = 0;
        for (let i = 0; i < pcmBuffers.length; i++) {
            merged.set(pcmBuffers[i], offset);
            offset += pcmBuffers[i].length;
        }

        if (totalSamples > 0) {
            const wavBlob = encodeWAV(merged, sampleRate);
            await processAudioIntake(wavBlob, 'recording.wav', liveVoiceTranscript);
        }
    }

    if (micRecordBtn) {
        micRecordBtn.addEventListener('click', () => {
            if (isRecording) stopRecording();
            else startRecording();
        });
    }

    if (uploadAudioBtn && audioFileInput) {
        uploadAudioBtn.addEventListener('click', () => {
            audioFileInput.click();
        });

        audioFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await processAudioIntake(file, file.name);
            audioFileInput.value = '';
        });
    }

    // Toggle Context Bar
    if (toggleContextBtn && contextBar) {
        toggleContextBtn.addEventListener('click', () => {
            const isHidden = contextBar.style.display === 'none';
            contextBar.style.display = isHidden ? 'block' : 'none';
        });
    }

    if (ctxMoodInput && ctxMoodVal) {
        ctxMoodInput.addEventListener('input', (e) => {
            ctxMoodVal.textContent = e.target.value;
        });
    }

    // Save Context Check-in
    if (saveContextBtn) {
        saveContextBtn.addEventListener('click', async () => {
            const sleepHours = parseFloat(ctxSleepInput.value) || 7.0;
            const workload = ctxWorkloadSelect.value || 'normal';
            const moodScore = parseInt(ctxMoodInput.value) || 7;

            const res = await window.API.saveContext({
                sleep_hours: sleepHours,
                workload: workload,
                mood_score: moodScore
            });

            if (res && res.status === 'success') {
                activeContext = res.context;
                if (headerContextSummary) {
                    headerContextSummary.textContent = `${activeContext.sleep_hours}h • ${activeContext.workload.toUpperCase()} • Mood ${activeContext.mood_score}`;
                }
                const baseRes = await window.API.getBaseline();
                if (baseRes && baseRes.baseline) {
                    activeBaseline = baseRes.baseline;
                    renderBaselineTelemetry(activeBaseline);
                }
                contextBar.style.display = 'none';
            }
        });
    }

    // 7-Day Trends Modal Handlers (Phase 6)
    if (toggleTrendsBtn && trendsModal) {
        toggleTrendsBtn.addEventListener('click', async () => {
            trendsModal.style.display = 'flex';
            const trendsRes = await window.API.getTrends();
            if (trendsRes && trendsRes.trends && trendsChartContainer) {
                trendsChartContainer.innerHTML = trendsRes.trends.map(t => {
                    const heightPct = Math.max(15, Math.min(100, t.energy_index));
                    const dayLabel = t.date ? t.date.split('-').slice(1).join('/') : '--';
                    return `
                        <div class="trend-day-col">
                            <span class="trend-score-label">${t.energy_index}%</span>
                            <div class="trend-bar-wrap">
                                <div class="trend-bar-fill" style="height:${heightPct}%;"></div>
                            </div>
                            <span class="trend-date-label">${dayLabel}</span>
                        </div>
                    `;
                }).join('');
            }
        });
    }

    if (closeTrendsBtn && trendsModal) {
        closeTrendsBtn.addEventListener('click', () => {
            trendsModal.style.display = 'none';
        });
    }

    /**
     * Tab switching inside drawer.
     */
    function switchDrawerTab(targetTab) {
        [tabBtnFusion, tabBtnBaseline, tabBtnVoice, tabBtnText, tabBtnPrompt, tabBtnJson].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });
        [drawerViewFusion, drawerViewBaseline, drawerViewVoice, drawerViewText, drawerViewPrompt, drawerViewJson].forEach(view => {
            if (view) {
                view.classList.remove('active');
                view.style.display = 'none';
            }
        });

        if (targetTab === 'fusion') {
            if (tabBtnFusion) tabBtnFusion.classList.add('active');
            if (drawerViewFusion) {
                drawerViewFusion.classList.add('active');
                drawerViewFusion.style.display = 'block';
            }
        } else if (targetTab === 'baseline') {
            if (tabBtnBaseline) tabBtnBaseline.classList.add('active');
            if (drawerViewBaseline) {
                drawerViewBaseline.classList.add('active');
                drawerViewBaseline.style.display = 'block';
            }
        } else if (targetTab === 'voice') {
            if (tabBtnVoice) tabBtnVoice.classList.add('active');
            if (drawerViewVoice) {
                drawerViewVoice.classList.add('active');
                drawerViewVoice.style.display = 'block';
            }
        } else if (targetTab === 'text') {
            if (tabBtnText) tabBtnText.classList.add('active');
            if (drawerViewText) {
                drawerViewText.classList.add('active');
                drawerViewText.style.display = 'block';
            }
        } else if (targetTab === 'prompt') {
            if (tabBtnPrompt) tabBtnPrompt.classList.add('active');
            if (drawerViewPrompt) {
                drawerViewPrompt.classList.add('active');
                drawerViewPrompt.style.display = 'block';
            }
        } else if (targetTab === 'json') {
            if (tabBtnJson) tabBtnJson.classList.add('active');
            if (drawerViewJson) {
                drawerViewJson.classList.add('active');
                drawerViewJson.style.display = 'block';
            }
        }
    }

    if (tabBtnFusion) tabBtnFusion.addEventListener('click', () => switchDrawerTab('fusion'));
    if (tabBtnBaseline) tabBtnBaseline.addEventListener('click', () => switchDrawerTab('baseline'));
    if (tabBtnVoice) tabBtnVoice.addEventListener('click', () => switchDrawerTab('voice'));
    if (tabBtnText) tabBtnText.addEventListener('click', () => switchDrawerTab('text'));
    if (tabBtnPrompt) tabBtnPrompt.addEventListener('click', () => switchDrawerTab('prompt'));
    if (tabBtnJson) tabBtnJson.addEventListener('click', () => switchDrawerTab('json'));

    /**
     * Initialize backend health, context, baselines, and model list.
     */
    async function initializeBackend() {
        if (statusDot) statusDot.className = 'status-dot connecting';
        if (statusText) statusText.textContent = 'Connecting to backend...';

        const statusRes = await window.API.getStatus();
        if (statusRes.error) {
            if (statusDot) statusDot.className = 'status-dot offline';
            if (statusText) statusText.textContent = 'Backend Offline';
            return;
        }

        const data = statusRes.data;
        const reasoning = data.reasoning_backend || {};

        if (reasoning.connected) {
            if (statusDot) statusDot.className = 'status-dot online';
            if (statusText) statusText.textContent = `Ollama Connected (${reasoning.available_models.length} models)`;
        } else {
            if (statusDot) statusDot.className = 'status-dot offline';
            if (statusText) statusText.textContent = 'Ollama Not Reachable';
        }

        if (backendLatency) {
            backendLatency.textContent = `Ping: ${statusRes.latencyMs} ms`;
        }

        if (modelSelect && reasoning.available_models && reasoning.available_models.length > 0) {
            modelSelect.innerHTML = '';
            reasoning.available_models.forEach(modelName => {
                const opt = document.createElement('option');
                opt.value = modelName;
                opt.textContent = modelName;
                if (modelName === reasoning.active_model) {
                    opt.selected = true;
                    currentModel = modelName;
                }
                modelSelect.appendChild(opt);
            });
            if (activeModelBadge) {
                activeModelBadge.textContent = currentModel;
            }
        }

        // Fetch active context & baseline
        const ctxRes = await window.API.getContext();
        if (ctxRes && ctxRes.context) {
            activeContext = ctxRes.context;
            if (headerContextSummary) {
                headerContextSummary.textContent = `${activeContext.sleep_hours}h • ${activeContext.workload.toUpperCase()} • Mood ${activeContext.mood_score}`;
            }
            if (ctxSleepInput) ctxSleepInput.value = activeContext.sleep_hours;
            if (ctxWorkloadSelect) ctxWorkloadSelect.value = activeContext.workload;
            if (ctxMoodInput) {
                ctxMoodInput.value = activeContext.mood_score;
                if (ctxMoodVal) ctxMoodVal.textContent = activeContext.mood_score;
            }
        }

        const baseRes = await window.API.getBaseline();
        if (baseRes && baseRes.baseline) {
            activeBaseline = baseRes.baseline;
            renderBaselineTelemetry(activeBaseline);
        }

        updatePromptAndSchemaViews(buildFrequencyInput("I'm feeling energized and ready to tackle my work."));
    }

    // Toggle Schema Drawer
    if (toggleSchemaBtn && schemaDrawer) {
        toggleSchemaBtn.addEventListener('click', () => {
            const isHidden = schemaDrawer.style.display === 'none';
            schemaDrawer.style.display = isHidden ? 'block' : 'none';
        });
    }

    if (closeSchemaBtn && schemaDrawer) {
        closeSchemaBtn.addEventListener('click', () => {
            schemaDrawer.style.display = 'none';
        });
    }

    // Auto-expand textarea on input
    if (userInput) {
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
        });

        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }

    // Form submission
    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleSendMessage();
        });
    }

    // Suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt) {
                handleSendMessage(prompt);
            }
        });
    });

    // Model selection change
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            currentModel = e.target.value;
            if (activeModelBadge) {
                activeModelBadge.textContent = currentModel;
            }
        });
    }

    // Clear Chat / New Session
    function resetChat() {
        const messages = chatMessages.querySelectorAll('.message-row');
        messages.forEach(msg => msg.remove());
        if (welcomeHero) {
            welcomeHero.style.display = 'flex';
        }
        updatePromptAndSchemaViews(buildFrequencyInput("Sample reflection or query..."));
        if (userInput) userInput.focus();
    }

    if (newChatBtn) newChatBtn.addEventListener('click', resetChat);
    if (clearChatBtn) clearChatBtn.addEventListener('click', resetChat);

    // Initial setup
    initializeBackend();
});
