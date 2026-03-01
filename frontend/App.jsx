/**
 * EmpathEase Frontend
 * ===================
 * Four screens: Landing → Session → Crisis (if needed) → Summary
 *
 * Chromatherapy: background tint shifts subtly based on detected emotion.
 * The UI responds with the antidote color — not a mirror of distress,
 * but a gentle guide toward calm. Transitions take 4 seconds so the
 * shift is felt, not consciously noticed.
 *
 * WebSocket: connects to ws://host:8000/ws/session/{sessionId}
 * Sends:    { text?, audio_base64?, image_base64?, is_voice_session? }
 * Receives: { text, fused_state, crisis_level, audio_base64? }
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ─────────────────────────────────────────────────────────────────────
// CHROMATHERAPY SYSTEM
// Antidote colors — guides user FROM distress TOWARD calm
// ─────────────────────────────────────────────────────────────────────

const CHROMA = {
  fear: { tint: "91, 143, 168", accent: "#5b8fa8", label: "Calming", bg: "10, 22, 40" },
  anger: { tint: "90, 158, 122", accent: "#5a9e7a", label: "Grounding", bg: "10, 26, 18" },
  sadness: { tint: "196, 149, 106", accent: "#c4956a", label: "Warming", bg: "26, 18, 8" },
  suppressed: { tint: "139, 126, 200", accent: "#8b7ec8", label: "Processing", bg: "16, 10, 26" },
  disgust: { tint: "106, 170, 138", accent: "#6aaa8a", label: "Cleansing", bg: "10, 26, 20" },
  surprise: { tint: "78,  205, 196", accent: "#4ecdc4", label: "Settling", bg: "10, 22, 24" },
  joy: { tint: "199, 165, 110", accent: "#c7a56e", label: "Nurturing", bg: "24, 20, 10" },
  neutral: { tint: "78,  205, 196", accent: "#4ecdc4", label: "Balanced", bg: "14, 22, 24" },
  crisis: { tint: "212, 160, 160", accent: "#d4a0a0", label: "Safe", bg: "26, 14, 14" },
};

const getChroma = (emotion, isCrisis) => {
  if (isCrisis) return CHROMA.crisis;
  return CHROMA[emotion] || CHROMA.neutral;
};

// ─────────────────────────────────────────────────────────────────────
// GLOBAL STYLES
// ─────────────────────────────────────────────────────────────────────

const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=JetBrains+Mono:wght@400;500&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:          #0e1214;
      --bg-surface:  #141a1c;
      --bg-card:     #1c2426;
      --bg-input:    #222c2e;
      --border:      rgba(255,255,255,0.06);
      --text:        #eef2f0;
      --text-muted:  #7a8e8a;
      --text-faint:  #3a4a48;
      --chroma-r:    78;
      --chroma-g:    205;
      --chroma-b:    196;
      --accent:      #4ecdc4;
      --font-display:'Cormorant Garamond', Georgia, serif;
      --font-body:   'DM Sans', system-ui, sans-serif;
      --font-mono:   'JetBrains Mono', monospace;
      --radius:      14px;
      --radius-sm:   8px;
      --transition-chroma: background 4s ease, box-shadow 4s ease;
    }

    html, body, #root {
      height: 100%; width: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      font-size: 15px;
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
      overflow: hidden;
    }

    ::selection { background: rgba(78,205,196,0.2); color: #4ecdc4; }
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--text-faint); border-radius: 2px; }

    @keyframes fade-up {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    @keyframes breathe {
      0%, 100% { transform: scale(1);    opacity: 0.5; }
      50%       { transform: scale(1.08); opacity: 0.9; }
    }
    @keyframes orbit {
      from { transform: rotate(0deg)   translateX(48px) rotate(0deg);   }
      to   { transform: rotate(360deg) translateX(48px) rotate(-360deg); }
    }
    @keyframes bar-dance {
      0%, 100% { transform: scaleY(0.3); }
      50%       { transform: scaleY(1.0); }
    }
    @keyframes crisis-breathe {
      0%, 100% { box-shadow: 0 0 0 0   rgba(212,160,160,0.3); }
      50%       { box-shadow: 0 0 0 20px rgba(212,160,160,0); }
    }
    @keyframes slide-message {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes chroma-shift {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
  `}</style>
);

// ─────────────────────────────────────────────────────────────────────
// HOOKS
// ─────────────────────────────────────────────────────────────────────

function useWebSocket(sessionId, onMessage) {
  const ws = useRef(null);
  const [status, setStatus] = useState("disconnected"); // disconnected|connecting|connected|error

  const connect = useCallback(() => {
    if (!sessionId) return;
    setStatus("connecting");
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws/session/${sessionId}`;
    ws.current = new WebSocket(url);

    ws.current.onopen = () => setStatus("connected");
    ws.current.onclose = () => { setStatus("disconnected"); };
    ws.current.onerror = () => setStatus("error");
    ws.current.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); } catch { }
    };
  }, [sessionId, onMessage]);

  const send = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  const disconnect = useCallback(() => {
    ws.current?.close();
  }, []);

  useEffect(() => {
    connect();
    return () => ws.current?.close();
  }, [connect]);

  return { status, send, disconnect, reconnect: connect };
}

function useAudioRecorder() {
  const [recording, setRecording] = useState(false);
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunks.current = [];
      mediaRecorder.current.ondataavailable = (e) => chunks.current.push(e.data);
      mediaRecorder.current.start();
      setRecording(true);
    } catch (e) {
      console.error("Mic access failed:", e);
    }
  }, []);

  const stop = useCallback(() => new Promise((resolve) => {
    if (!mediaRecorder.current) { resolve(null); return; }
    mediaRecorder.current.onstop = () => {
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      const reader = new FileReader();
      reader.onloadend = () => {
        const b64 = reader.result.split(",")[1];
        // Stop all tracks
        mediaRecorder.current.stream.getTracks().forEach(t => t.stop());
        setRecording(false);
        resolve(b64);
      };
      reader.readAsDataURL(blob);
    };
    mediaRecorder.current.stop();
  }), []);

  return { recording, start, stop };
}

function useWebcam() {
  const videoRef = useRef(null);
  const [active, setActive] = useState(false);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setActive(true);
      }
    } catch (e) {
      console.error("Camera access failed:", e);
    }
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !active) return null;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 320;
    canvas.height = videoRef.current.videoHeight || 240;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
  }, [active]);

  const stop = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      setActive(false);
    }
  }, []);

  return { videoRef, active, start, stop, captureFrame };
}

// ─────────────────────────────────────────────────────────────────────
// CHROMATHERAPY BACKGROUND
// ─────────────────────────────────────────────────────────────────────

const ChromaBackground = ({ emotion, isCrisis }) => {
  const chroma = getChroma(emotion, isCrisis);
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 0,
      transition: "background 4s ease",
      background: `
        radial-gradient(ellipse 120% 80% at 20% 20%,
          rgba(${chroma.tint}, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 80% 120% at 80% 80%,
          rgba(${chroma.tint}, 0.05) 0%, transparent 60%),
        #${chroma.bg.split(", ").map(v => parseInt(v).toString(16).padStart(2, "0")).join("")}
      `,
      pointerEvents: "none",
    }} />
  );
};

// ─────────────────────────────────────────────────────────────────────
// ATOMS
// ─────────────────────────────────────────────────────────────────────

const Logo = ({ size = "md" }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
    <div style={{
      width: size === "lg" ? 48 : 32,
      height: size === "lg" ? 48 : 32,
      borderRadius: "50%",
      background: "radial-gradient(circle at 40% 40%, #4ecdc4, #2a8a85)",
      boxShadow: "0 0 24px rgba(78,205,196,0.3)",
      flexShrink: 0,
    }} />
    <span style={{
      fontFamily: "var(--font-display)",
      fontSize: size === "lg" ? "2rem" : "1.25rem",
      fontWeight: 400,
      letterSpacing: "0.02em",
      color: "var(--text)",
    }}>
      Empath<span style={{ color: "#4ecdc4", fontStyle: "italic" }}>Ease</span>
    </span>
  </div>
);

const ConnectionDot = ({ status }) => {
  const colors = {
    connected: "#4ecdc4",
    connecting: "#c7a56e",
    disconnected: "#7a8e8a",
    error: "#e07070",
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 7, height: 7, borderRadius: "50%",
        background: colors[status] || colors.disconnected,
        boxShadow: status === "connected" ? `0 0 6px ${colors.connected}` : "none",
        animation: status === "connecting" ? "breathe 1.2s ease infinite" : "none",
      }} />
      <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        {status}
      </span>
    </div>
  );
};

const EmotionBadge = ({ emotion, confidence, chroma }) => (
  <div style={{
    display: "flex", alignItems: "center", gap: 8,
    padding: "6px 12px",
    background: `rgba(${chroma.tint}, 0.12)`,
    border: `1px solid rgba(${chroma.tint}, 0.2)`,
    borderRadius: 20,
    transition: "all 4s ease",
  }}>
    <div style={{
      width: 6, height: 6, borderRadius: "50%",
      background: chroma.accent,
      animation: "breathe 3s ease infinite",
    }} />
    <span style={{
      fontFamily: "var(--font-mono)",
      fontSize: 12,
      color: chroma.accent,
      textTransform: "lowercase",
      transition: "color 4s ease",
    }}>
      {emotion}
    </span>
    {confidence && (
      <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        {Math.round(confidence * 100)}%
      </span>
    )}
  </div>
);

const ValenceArousalGauge = ({ valence, arousal, chroma }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
    {[
      { label: "Valence", value: (valence + 1) / 2, raw: valence, signed: true },
      { label: "Arousal", value: arousal, raw: arousal, signed: false },
    ].map(({ label, value, raw, signed }) => (
      <div key={label}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {label}
          </span>
          <span style={{ fontSize: 11, color: chroma.accent, fontFamily: "var(--font-mono)", transition: "color 4s ease" }}>
            {signed ? (raw > 0 ? `+${raw.toFixed(2)}` : raw.toFixed(2)) : raw.toFixed(2)}
          </span>
        </div>
        <div style={{
          height: 3, background: "var(--text-faint)", borderRadius: 2, overflow: "hidden",
        }}>
          <div style={{
            height: "100%", width: `${Math.max(2, value * 100)}%`,
            background: `linear-gradient(90deg, rgba(${chroma.tint},0.4), ${chroma.accent})`,
            borderRadius: 2,
            transition: "width 1.5s ease, background 4s ease",
          }} />
        </div>
      </div>
    ))}
  </div>
);

const Waveform = ({ active }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 3, height: 24 }}>
    {[...Array(7)].map((_, i) => (
      <div key={i} style={{
        width: 3, height: "100%",
        background: "#4ecdc4",
        borderRadius: 2,
        transformOrigin: "bottom",
        transform: active ? "scaleY(1)" : "scaleY(0.2)",
        animation: active ? `bar-dance ${0.5 + i * 0.08}s ease infinite` : "none",
        animationDelay: active ? `${i * 0.07}s` : "0s",
        opacity: active ? 1 : 0.3,
        transition: "opacity 0.3s ease",
      }} />
    ))}
  </div>
);

const ChatMessage = ({ message, isUser, chroma }) => (
  <div style={{
    display: "flex",
    justifyContent: isUser ? "flex-end" : "flex-start",
    animation: "slide-message 0.3s ease",
    marginBottom: 16,
  }}>
    {!isUser && (
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: `radial-gradient(circle at 40% 40%, ${chroma.accent}, rgba(${chroma.tint},0.3))`,
        flexShrink: 0, marginRight: 10, marginTop: 4,
        transition: "background 4s ease",
      }} />
    )}
    <div style={{
      maxWidth: "72%",
      padding: "12px 16px",
      borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
      background: isUser
        ? "rgba(255,255,255,0.07)"
        : `rgba(${chroma.tint}, 0.09)`,
      border: isUser
        ? "1px solid rgba(255,255,255,0.08)"
        : `1px solid rgba(${chroma.tint}, 0.15)`,
      color: "var(--text)",
      fontSize: 14,
      lineHeight: 1.65,
      transition: "background 4s ease, border-color 4s ease",
    }}>
      {message.text}
      <div style={{
        marginTop: 6,
        fontSize: 11,
        color: "var(--text-muted)",
        fontFamily: "var(--font-mono)",
        textAlign: isUser ? "right" : "left",
      }}>
        {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────
// SCREEN 1 — LANDING
// ─────────────────────────────────────────────────────────────────────

const LandingScreen = ({ onStart }) => {
  const [consented, setConsented] = useState(false);
  const [hovering, setHovering] = useState(false);

  return (
    <div style={{
      position: "relative", zIndex: 1,
      height: "100vh", display: "flex",
      flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: 40,
    }}>
      {/* Orbital decoration */}
      <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", pointerEvents: "none" }}>
        {[80, 130, 190].map((r, i) => (
          <div key={i} style={{
            position: "absolute",
            width: r * 2, height: r * 2,
            top: -r, left: -r,
            borderRadius: "50%",
            border: `1px solid rgba(78,205,196,${0.06 - i * 0.015})`,
            animation: `breathe ${4 + i * 1.5}s ease infinite`,
            animationDelay: `${i * 0.8}s`,
          }} />
        ))}
        <div style={{
          position: "absolute",
          width: 6, height: 6, borderRadius: "50%",
          background: "#4ecdc4", opacity: 0.6,
          top: -3, left: -3,
          animation: `orbit ${12}s linear infinite`,
        }} />
        <div style={{
          position: "absolute",
          width: 4, height: 4, borderRadius: "50%",
          background: "#c7a56e", opacity: 0.5,
          top: -2, left: -2,
          animation: `orbit ${20}s linear infinite reverse`,
          "--orbit-r": "80px",
        }} />
      </div>

      {/* Content */}
      <div style={{ position: "relative", zIndex: 1, textAlign: "center", maxWidth: 520 }}>
        <div style={{ marginBottom: 32, animation: "fade-up 0.8s ease" }}>
          <Logo size="lg" />
        </div>

        <h1 style={{
          fontFamily: "var(--font-display)",
          fontSize: "clamp(1.6rem, 4vw, 2.4rem)",
          fontWeight: 300,
          lineHeight: 1.3,
          color: "var(--text)",
          marginBottom: 16,
          animation: "fade-up 0.8s ease 0.1s both",
        }}>
          A space to be heard,
          <br />
          <em style={{ color: "#4ecdc4" }}>without judgment.</em>
        </h1>

        <p style={{
          color: "var(--text-muted)",
          fontSize: 15,
          lineHeight: 1.7,
          marginBottom: 36,
          animation: "fade-up 0.8s ease 0.2s both",
        }}>
          EmpathEase is a supportive companion for your emotional wellbeing.
          It listens, reflects, and gently helps you understand what you're feeling.
          <br />
          <span style={{ fontSize: 13, color: "var(--text-faint)", marginTop: 8, display: "block" }}>
            Not a replacement for professional care — a bridge to yourself.
          </span>
        </p>

        {/* Consent */}
        <div style={{
          padding: "16px 20px",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          marginBottom: 28,
          textAlign: "left",
          animation: "fade-up 0.8s ease 0.3s both",
        }}>
          <label style={{ display: "flex", gap: 12, cursor: "pointer", alignItems: "flex-start" }}>
            <div
              onClick={() => setConsented(!consented)}
              style={{
                width: 18, height: 18, borderRadius: 4,
                border: `1.5px solid ${consented ? "#4ecdc4" : "var(--text-faint)"}`,
                background: consented ? "rgba(78,205,196,0.2)" : "transparent",
                flexShrink: 0, marginTop: 2,
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "all 0.2s ease", cursor: "pointer",
              }}
            >
              {consented && <span style={{ color: "#4ecdc4", fontSize: 12, lineHeight: 1 }}>✓</span>}
            </div>
            <span style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>
              I understand this is a supportive tool, not a licensed therapist.
              My session data is processed privately and not shared.
              In a crisis, I will contact a professional.
            </span>
          </label>
        </div>

        {/* Crisis resources */}
        <div style={{
          fontSize: 12, color: "var(--text-muted)",
          marginBottom: 28,
          animation: "fade-up 0.8s ease 0.35s both",
        }}>
          If you're in crisis right now:{" "}
          <span style={{ color: "#c4956a", fontFamily: "var(--font-mono)" }}>iCall — 9152987821</span>
          {" "}·{" "}
          <span style={{ color: "#c4956a", fontFamily: "var(--font-mono)" }}>Vandrevala — 1860-2662-345</span>
        </div>

        <button
          disabled={!consented}
          onMouseEnter={() => setHovering(true)}
          onMouseLeave={() => setHovering(false)}
          onClick={() => consented && onStart()}
          style={{
            padding: "14px 40px",
            background: consented
              ? (hovering ? "rgba(78,205,196,0.2)" : "rgba(78,205,196,0.12)")
              : "rgba(255,255,255,0.03)",
            border: `1px solid ${consented ? "rgba(78,205,196,0.4)" : "var(--border)"}`,
            borderRadius: 40,
            color: consented ? "#4ecdc4" : "var(--text-faint)",
            fontFamily: "var(--font-body)",
            fontSize: 15,
            fontWeight: 500,
            cursor: consented ? "pointer" : "not-allowed",
            transition: "all 0.3s ease",
            letterSpacing: "0.02em",
            animation: "fade-up 0.8s ease 0.4s both",
          }}
        >
          Begin session
        </button>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// SCREEN 2 — SESSION
// ─────────────────────────────────────────────────────────────────────

const SessionScreen = ({ sessionId, onCrisis, onEnd, onEmotionChange }) => {
  const [messages, setMessages] = useState([
    {
      text: "Hello. Main AUM hoon... kya main aapka naam jaan sakta hoon ya koi Nickname bhi chalega.",
      isUser: false,
      timestamp: Date.now(),
      emotion: "neutral",
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [fusedState, setFusedState] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [sessionStart] = useState(Date.now());
  const [isVoice, setIsVoice] = useState(false);

  const chatEnd = useRef(null);
  const inputRef = useRef(null);
  const frameTimer = useRef(null);
  const audioCtxRef = useRef(null);
  const lastWasVoiceRef = useRef(false); // tracks if last send was a voice (mic) turn

  const { recording, start: startRec, stop: stopRec } = useAudioRecorder();
  const { videoRef, active: camActive, start: startCam, captureFrame } = useWebcam();

  const emotion = fusedState?.dominant_emotion || "neutral";
  const chroma = getChroma(emotion, false);

  // ── TTS audio playback via AudioContext (bypasses browser autoplay policy) ──
  const playTTSAudio = useCallback((b64) => {
    try {
      // Create/reuse a shared AudioContext
      if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;

      // Resume if suspended (needed the first time, before any user gesture)
      const doPlay = () => {
        // Decode base64 → ArrayBuffer
        const raw = atob(b64);
        const buf = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);

        ctx.decodeAudioData(buf.buffer, (decoded) => {
          const src = ctx.createBufferSource();
          src.buffer = decoded;
          src.connect(ctx.destination);
          src.start(0);
        }, (err) => console.warn("TTS decode failed:", err));
      };

      if (ctx.state === "suspended") {
        ctx.resume().then(doPlay).catch(console.warn);
      } else {
        doPlay();
      }
    } catch (err) {
      console.warn("TTS playback failed:", err);
    }
  }, []);

  const handleMessage = useCallback((data) => {
    setThinking(false);

    // Lightweight FER-only state update — MERGE into existing state, don't replace.
    // FER frames only know about the face modality; a full replace would wipe out
    // text_emotion / vocal_emotion / valence / arousal from the last full turn.
    if (data.type === "state_update") {
      if (data.fused_state) {
        setFusedState(prev => prev ? {
          ...prev,
          // Update only face-related fields from the live FER frame
          facial_emotion: data.fused_state.facial_emotion ?? prev.facial_emotion,
          // If FER has a stronger signal than current dominant, blend it in
          dominant_emotion: data.fused_state.dominant_emotion ?? prev.dominant_emotion,
          confidence: data.fused_state.confidence ?? prev.confidence,
          // Preserve full-turn fields so the meters don't blank out
          text_emotion: prev.text_emotion,
          vocal_emotion: prev.vocal_emotion,
          valence: prev.valence,
          arousal: prev.arousal,
          incongruence: prev.incongruence,
          crisis_level: prev.crisis_level,
        } : data.fused_state);
        onEmotionChange?.(data.fused_state.dominant_emotion);
      }
      return;
    }

    if (data.crisis_level === "urgent") {
      onCrisis(data);
      return;
    }
    if (data.fused_state) {
      setFusedState(data.fused_state);
      onEmotionChange?.(data.fused_state.dominant_emotion);
    }

    // Show voice transcription as user bubble — but ONLY for voice turns.
    // Typed text is already added optimistically by sendMessage(), skip the echo.
    if (data.user_text && lastWasVoiceRef.current) {
      setMessages(prev => [...prev, {
        text: `🎤 ${data.user_text}`, isUser: true,
        timestamp: Date.now() - 1,  // just before the AI response
      }]);
    }

    if (data.text) {
      setMessages(prev => [...prev, {
        text: data.text, isUser: false,
        timestamp: Date.now(),
        emotion: data.fused_state?.dominant_emotion,
      }]);
    }
    if (data.audio_base64) {
      playTTSAudio(data.audio_base64);
    }
  }, [onCrisis, onEmotionChange]);

  const { status, send } = useWebSocket(sessionId, handleMessage);

  // Capture webcam frame periodically and send for FER
  useEffect(() => {
    if (!camActive) return;
    frameTimer.current = setInterval(() => {
      const frame = captureFrame();
      if (frame && status === "connected") {
        send({ image_base64: frame });
      }
    }, 2500);
    return () => clearInterval(frameTimer.current);
  }, [camActive, captureFrame, send, status]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    startCam();
  }, [startCam]);

  const sendMessage = useCallback(async (text, audioB64 = null) => {
    if (!text?.trim() && !audioB64) return;
    const isVoiceTurn = !!audioB64;
    lastWasVoiceRef.current = isVoiceTurn;
    const payload = {
      text: text?.trim() || undefined,
      audio_base64: audioB64 || undefined,
      image_base64: captureFrame() || undefined,
      is_voice_session: isVoice,
    };
    if (text?.trim()) {
      // Only add typed message optimistically (voice transcription comes back from server)
      setMessages(prev => [...prev, {
        text: text.trim(), isUser: true, timestamp: Date.now(),
      }]);
    }
    setInputText("");
    setThinking(true);
    const sent = send(payload);
    if (!sent) {
      setThinking(false);
      setMessages(prev => [...prev, {
        text: "Connection lost. Please check your internet and try again.",
        isUser: false, timestamp: Date.now(), isError: true,
      }]);
    }
  }, [send, captureFrame, isVoice]);

  const handleMicDown = useCallback(async () => {
    if (recording) return; // already recording
    // Resume AudioContext on user gesture so TTS can play later without being blocked
    if (audioCtxRef.current?.state === "suspended") {
      audioCtxRef.current.resume().catch(() => { });
    } else if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    setIsVoice(true);
    startRec();
  }, [recording, startRec]);

  const handleMicUp = useCallback(async () => {
    if (!recording) return; // nothing to stop
    const b64 = await stopRec();
    if (b64) sendMessage(null, b64);
  }, [recording, stopRec, sendMessage]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputText);
    }
  };

  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const elapsedStr = `${Math.floor(elapsed / 60).toString().padStart(2, "0")}:${(elapsed % 60).toString().padStart(2, "0")}`;

  return (
    <div style={{
      position: "relative", zIndex: 1,
      height: "100vh", display: "flex", flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "14px 24px",
        borderBottom: "1px solid var(--border)",
        background: "rgba(14,18,20,0.8)",
        backdropFilter: "blur(12px)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <Logo />
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {fusedState && (
            <EmotionBadge
              emotion={emotion}
              confidence={fusedState.confidence}
              chroma={chroma}
            />
          )}
          {/* Chromatherapy indicator */}
          {fusedState && (
            <div style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "4px 10px",
              background: `rgba(${chroma.tint}, 0.08)`,
              borderRadius: 20,
              transition: "all 4s ease",
            }}>
              <div style={{
                width: 5, height: 5, borderRadius: "50%",
                background: chroma.accent,
                transition: "background 4s ease",
              }} />
              <span style={{
                fontSize: 11, color: chroma.accent,
                fontFamily: "var(--font-mono)",
                transition: "color 4s ease",
              }}>
                {chroma.label}
              </span>
            </div>
          )}
          <ConnectionDot status={status} />
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {elapsedStr}
          </span>
          <button
            onClick={onEnd}
            style={{
              padding: "5px 14px",
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: 20,
              color: "var(--text-muted)",
              fontSize: 12, cursor: "pointer",
              fontFamily: "var(--font-body)",
              transition: "all 0.2s ease",
            }}
          >
            End session
          </button>
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Left — Webcam + Live AI Metrics */}
        <div style={{
          width: 300, flexShrink: 0,
          borderRight: "1px solid var(--border)",
          display: "flex", flexDirection: "column",
          padding: 16, gap: 12, overflowY: "auto",
          background: "rgba(10,14,16,0.4)",
          backdropFilter: "blur(8px)",
        }}>
          {/* Webcam */}
          <div style={{ position: "relative" }}>
            <div style={{
              borderRadius: "var(--radius)",
              overflow: "hidden",
              background: "var(--bg-card)",
              aspectRatio: "4/3",
              display: "flex", alignItems: "center", justifyContent: "center",
              border: `1px solid rgba(${chroma.tint}, 0.25)`,
              boxShadow: `0 0 20px rgba(${chroma.tint}, 0.08)`,
              transition: "border-color 4s ease, box-shadow 4s ease",
            }}>
              <video
                ref={videoRef}
                muted
                style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
              />
              {!camActive && (
                <div style={{ position: "absolute", textAlign: "center" }}>
                  <div style={{ fontSize: 32, marginBottom: 8 }}>📷</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>Camera off</div>
                </div>
              )}
            </div>
            {/* Live FER overlay */}
            {camActive && (
              <div style={{
                position: "absolute", bottom: 8, left: 8, right: 8,
                padding: "6px 10px",
                background: `rgba(${chroma.tint}, 0.18)`,
                backdropFilter: "blur(8px)",
                borderRadius: "var(--radius-sm)",
                border: `1px solid rgba(${chroma.tint}, 0.25)`,
                transition: "all 4s ease",
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span style={{ fontSize: 11, color: chroma.accent, fontFamily: "var(--font-mono)", transition: "color 4s ease" }}>
                  FER: {fusedState?.facial_emotion || "scanning..."}
                </span>
                <span style={{
                  fontSize: 10, padding: "2px 6px",
                  background: `rgba(${chroma.tint}, 0.2)`,
                  borderRadius: 10, color: chroma.accent,
                  fontFamily: "var(--font-mono)",
                }}>
                  LIVE
                </span>
              </div>
            )}
          </div>

          {/* Fusion Engine Panel */}
          <div style={{
            padding: "12px 14px",
            background: `rgba(${chroma.tint}, 0.05)`,
            border: `1px solid rgba(${chroma.tint}, 0.12)`,
            borderRadius: "var(--radius)",
            transition: "all 4s ease",
          }}>
            <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 10, letterSpacing: "0.1em" }}>
              ◈ FUSION ENGINE
            </div>

            {/* Face modality */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>FACE</span>
                <span style={{ fontSize: 11, color: fusedState?.facial_emotion ? chroma.accent : "var(--text-faint)", fontFamily: "var(--font-mono)" }}>
                  {fusedState?.facial_emotion || (fusedState ? "—" : "scanning")}
                </span>
              </div>
              <div style={{ height: 2, background: "var(--text-faint)", borderRadius: 2 }}>
                <div style={{
                  height: "100%",
                  width: fusedState?.facial_emotion
                    ? `${Math.round((fusedState.emotions?.[fusedState.facial_emotion] ?? 0.5) * 100)}%`
                    : "0%",
                  background: `linear-gradient(90deg, rgba(${chroma.tint},0.4), ${chroma.accent})`,
                  borderRadius: 2, transition: "width 1s ease, background 4s ease"
                }} />
              </div>
            </div>

            {/* Text modality */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>TEXT</span>
                <span style={{ fontSize: 11, color: fusedState?.text_emotion ? chroma.accent : "var(--text-faint)", fontFamily: "var(--font-mono)" }}>
                  {fusedState?.text_emotion || (fusedState ? "—" : "waiting")}
                </span>
              </div>
              <div style={{ height: 2, background: "var(--text-faint)", borderRadius: 2 }}>
                <div style={{
                  height: "100%",
                  width: fusedState?.text_emotion
                    ? `${Math.round((fusedState.emotions?.[fusedState.text_emotion] ?? 0.6) * 100)}%`
                    : "0%",
                  background: `linear-gradient(90deg, rgba(${chroma.tint},0.4), ${chroma.accent})`,
                  borderRadius: 2, transition: "width 1s ease, background 4s ease"
                }} />
              </div>
            </div>

            {/* Vocal modality — only active on voice turns */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>VOCAL</span>
                <span style={{
                  fontSize: 11, fontFamily: "var(--font-mono)",
                  color: fusedState?.vocal_emotion ? chroma.accent
                    : fusedState ? "var(--text-faint)" : "var(--text-faint)",
                  fontStyle: fusedState && !fusedState.vocal_emotion ? "italic" : "normal",
                }}>
                  {fusedState?.vocal_emotion || (fusedState ? "mic only" : "waiting")}
                </span>
              </div>
              <div style={{ height: 2, background: "var(--text-faint)", borderRadius: 2 }}>
                <div style={{
                  height: "100%",
                  width: fusedState?.vocal_emotion
                    ? `${Math.round((fusedState.emotions?.[fusedState.vocal_emotion] ?? 0.5) * 100)}%`
                    : "0%",
                  background: `linear-gradient(90deg, rgba(${chroma.tint},0.4), ${chroma.accent})`,
                  borderRadius: 2, transition: "width 1s ease, background 4s ease"
                }} />
              </div>
            </div>

            {/* Fused dominant */}
            <div style={{
              padding: "8px 10px",
              background: `rgba(${chroma.tint}, 0.1)`,
              borderRadius: "var(--radius-sm)",
              display: "flex", justifyContent: "space-between", alignItems: "center",
              transition: "background 4s ease",
            }}>
              <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>FUSED</span>
              <span style={{ fontSize: 13, color: chroma.accent, fontFamily: "var(--font-mono)", fontWeight: 500, transition: "color 4s ease" }}>
                {emotion} {fusedState ? `· ${Math.round((fusedState.confidence || 0) * 100)}%` : ""}
              </span>
            </div>
          </div>

          {/* Valence / Arousal */}
          {fusedState && (
            <div style={{
              padding: "12px 14px",
              background: "rgba(255,255,255,0.02)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
            }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 10, letterSpacing: "0.1em" }}>
                ◈ AFFECT STATE
              </div>
              <ValenceArousalGauge valence={fusedState.valence || 0} arousal={fusedState.arousal || 0} chroma={chroma} />
            </div>
          )}

          {/* Suppression Detection */}
          {fusedState?.incongruence?.detected && (
            <div style={{
              padding: "10px 14px",
              background: "rgba(139,126,200,0.1)",
              border: "1px solid rgba(139,126,200,0.25)",
              borderRadius: "var(--radius-sm)",
              animation: "fade-up 0.3s ease",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#8b7ec8", animation: "breathe 1.5s ease infinite" }} />
                <span style={{ fontSize: 10, color: "#8b7ec8", fontFamily: "var(--font-mono)", letterSpacing: "0.1em" }}>SUPPRESSION DETECTED</span>
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                {fusedState.incongruence.details || "Emotional incongruence across modalities"}
              </div>
            </div>
          )}

          {/* Crisis Risk Meter */}
          <div style={{
            padding: "10px 14px",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
          }}>
            <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 8, letterSpacing: "0.1em" }}>◈ CRISIS MONITOR</div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ flex: 1, height: 4, background: "var(--text-faint)", borderRadius: 2 }}>
                <div style={{
                  height: "100%", borderRadius: 2,
                  width: fusedState?.crisis_level === "urgent" ? "100%"
                    : fusedState?.crisis_level === "moderate" ? "60%"
                      : fusedState?.crisis_level === "watch" ? "30%" : "5%",
                  background: fusedState?.crisis_level === "urgent" ? "linear-gradient(90deg,#e07070,#ff4444)"
                    : fusedState?.crisis_level === "moderate" ? "linear-gradient(90deg,#c4956a,#e07070)"
                      : `linear-gradient(90deg, rgba(${chroma.tint},0.4), ${chroma.accent})`,
                  transition: "width 1s ease, background 1s ease",
                }} />
              </div>
              <span style={{
                fontSize: 10, fontFamily: "var(--font-mono)",
                color: fusedState?.crisis_level === "urgent" ? "#e07070"
                  : fusedState?.crisis_level === "moderate" ? "#c4956a"
                    : "var(--text-faint)",
              }}>
                {fusedState?.crisis_level || "none"}
              </span>
            </div>
          </div>
        </div>


        {/* Center — Chat */}
        <div style={{
          flex: 1, display: "flex", flexDirection: "column", overflow: "hidden",
        }}>
          {/* Messages */}
          <div style={{
            flex: 1, overflowY: "auto",
            padding: "24px 28px",
          }}>
            {messages.length === 0 && (
              <div style={{
                height: "100%", display: "flex",
                flexDirection: "column", alignItems: "center", justifyContent: "center",
                gap: 12, animation: "fade-in 0.8s ease",
              }}>
                <div style={{
                  width: 52, height: 52, borderRadius: "50%",
                  background: `radial-gradient(circle at 40% 40%, ${chroma.accent}, rgba(${chroma.tint},0.2))`,
                  animation: "breathe 4s ease infinite",
                  transition: "background 4s ease",
                }} />
                <p style={{
                  fontFamily: "var(--font-display)",
                  fontSize: "1.3rem",
                  fontWeight: 300,
                  color: "var(--text-muted)",
                  textAlign: "center",
                  lineHeight: 1.5,
                }}>
                  Aap kaise feel kar rahe hain aaj?
                  <br />
                  <span style={{ fontSize: "1rem", fontStyle: "italic" }}>
                    How are you feeling today?
                  </span>
                </p>
              </div>
            )}

            {messages.map((msg, i) => (
              <ChatMessage key={i} message={msg} isUser={msg.isUser} chroma={chroma} />
            ))}

            {thinking && (
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: "50%",
                  background: `radial-gradient(circle at 40% 40%, ${chroma.accent}, rgba(${chroma.tint},0.3))`,
                  transition: "background 4s ease",
                }} />
                <div style={{ display: "flex", gap: 6, padding: "12px 16px", background: `rgba(${chroma.tint}, 0.07)`, borderRadius: "4px 18px 18px 18px", transition: "background 4s ease" }}>
                  {[0.1, 0.25, 0.4].map((d, i) => (
                    <div key={i} style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: chroma.accent,
                      animation: `breathe 1.2s ease infinite`,
                      animationDelay: `${d}s`,
                      transition: "background 4s ease",
                    }} />
                  ))}
                </div>
              </div>
            )}
            <div ref={chatEnd} />
          </div>

          {/* Input */}
          <div style={{
            padding: "16px 24px",
            borderTop: "1px solid var(--border)",
            background: "rgba(14,18,20,0.6)",
            backdropFilter: "blur(12px)",
          }}>
            <div style={{
              display: "flex", gap: 10, alignItems: "flex-end",
            }}>
              {/* Mic button */}
              <button
                onMouseDown={handleMicDown}
                onMouseUp={handleMicUp}
                onMouseLeave={handleMicUp}
                onTouchStart={(e) => { e.preventDefault(); handleMicDown(); }}
                onTouchEnd={(e) => { e.preventDefault(); handleMicUp(); }}
                style={{
                  width: 44, height: 44, borderRadius: "50%", flexShrink: 0,
                  background: recording
                    ? "rgba(224,112,112,0.2)"
                    : `rgba(${chroma.tint}, 0.1)`,
                  border: `1.5px solid ${recording ? "#e07070" : `rgba(${chroma.tint}, 0.3)`}`,
                  color: recording ? "#e07070" : chroma.accent,
                  fontSize: 18, cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  transition: "all 0.3s ease",
                  animation: recording ? "crisis-breathe 1.5s ease infinite" : "none",
                }}
              >
                {recording ? "⏹" : "🎤"}
              </button>

              {recording && <Waveform active />}

              {!recording && (
                <>
                  <div style={{
                    flex: 1,
                    background: "var(--bg-input)",
                    border: "1px solid var(--border)",
                    borderRadius: 22,
                    padding: "10px 16px",
                    display: "flex", alignItems: "center",
                  }}>
                    <textarea
                      ref={inputRef}
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Type here or press the mic to speak..."
                      rows={1}
                      style={{
                        flex: 1, background: "transparent", border: "none", outline: "none",
                        color: "var(--text)", fontFamily: "var(--font-body)", fontSize: 14,
                        resize: "none", lineHeight: 1.5,
                      }}
                    />
                  </div>
                  <button
                    onClick={() => sendMessage(inputText)}
                    disabled={!inputText.trim() || thinking}
                    style={{
                      width: 44, height: 44, borderRadius: "50%", flexShrink: 0,
                      background: inputText.trim() && !thinking
                        ? `rgba(${chroma.tint}, 0.15)`
                        : "rgba(255,255,255,0.03)",
                      border: `1.5px solid ${inputText.trim() && !thinking ? `rgba(${chroma.tint}, 0.4)` : "var(--border)"}`,
                      color: inputText.trim() && !thinking ? chroma.accent : "var(--text-faint)",
                      fontSize: 16, cursor: inputText.trim() && !thinking ? "pointer" : "default",
                      transition: "all 0.3s ease",
                    }}
                  >
                    ↑
                  </button>
                </>
              )}
            </div>
            <div style={{
              marginTop: 8, fontSize: 11, color: "var(--text-faint)", textAlign: "center",
            }}>
              Not a licensed therapist · In crisis? Call iCall: 9152987821
            </div>
          </div>
        </div>
      </div>
    </div >
  );
};

// ─────────────────────────────────────────────────────────────────────
// SCREEN 3 — CRISIS
// ─────────────────────────────────────────────────────────────────────

const CrisisScreen = ({ data, onContinue }) => (
  <div style={{
    position: "relative", zIndex: 1,
    height: "100vh", display: "flex",
    flexDirection: "column", alignItems: "center", justifyContent: "center",
    padding: 40,
  }}>
    <div style={{
      maxWidth: 520, width: "100%",
      animation: "fade-up 0.6s ease",
    }}>
      <div style={{
        width: 64, height: 64, borderRadius: "50%",
        background: "rgba(212,160,160,0.15)",
        border: "1.5px solid rgba(212,160,160,0.3)",
        display: "flex", alignItems: "center", justifyContent: "center",
        margin: "0 auto 24px",
        animation: "crisis-breathe 3s ease infinite",
        fontSize: 28,
      }}>
        🤝
      </div>

      <h2 style={{
        fontFamily: "var(--font-display)",
        fontSize: "1.8rem", fontWeight: 400,
        textAlign: "center", marginBottom: 16,
        color: "var(--text)",
      }}>
        You are not alone.
      </h2>

      <p style={{
        color: "var(--text-muted)", textAlign: "center",
        lineHeight: 1.8, marginBottom: 32, fontSize: 15,
      }}>
        {data?.text || "Jo aapne share kiya, usse sun ke mujhe bahut chinta ho rahi hai. Aap ki safety abhi sabse important hai."}
      </p>

      {/* Resources */}
      <div style={{
        padding: 24,
        background: "rgba(212,160,160,0.06)",
        border: "1px solid rgba(212,160,160,0.15)",
        borderRadius: "var(--radius)",
        marginBottom: 24,
      }}>
        <div style={{ fontSize: 12, color: "#d4a0a0", fontFamily: "var(--font-mono)", marginBottom: 16, letterSpacing: "0.1em" }}>
          SUPPORT AVAILABLE NOW
        </div>
        {[
          { name: "Vandrevala Foundation", number: "1860-2662-345", note: "24/7, free" },
          { name: "iCall", number: "9152987821", note: "Mon–Sat, 8AM–10PM" },
          { name: "AASRA", number: "91-22-27546669", note: "24/7" },
          { name: "Emergency", number: "112", note: "" },
        ].map((r) => (
          <div key={r.name} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "10px 0",
            borderBottom: "1px solid rgba(255,255,255,0.04)",
          }}>
            <div>
              <div style={{ fontSize: 14, color: "var(--text)" }}>{r.name}</div>
              {r.note && <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{r.note}</div>}
            </div>
            <a
              href={`tel:${r.number.replace(/-/g, "")}`}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 14,
                color: "#d4a0a0",
                textDecoration: "none",
                padding: "4px 10px",
                background: "rgba(212,160,160,0.1)",
                borderRadius: 20,
              }}
            >
              {r.number}
            </a>
          </div>
        ))}
      </div>

      <div style={{ textAlign: "center" }}>
        <p style={{ fontSize: 14, color: "var(--text-muted)", marginBottom: 20 }}>
          Kya aap abhi safe hain?
          <br />
          <span style={{ fontSize: 13, fontStyle: "italic" }}>Are you safe right now?</span>
        </p>
        <button
          onClick={onContinue}
          style={{
            padding: "12px 32px",
            background: "rgba(212,160,160,0.1)",
            border: "1px solid rgba(212,160,160,0.25)",
            borderRadius: 30,
            color: "#d4a0a0",
            fontSize: 14, cursor: "pointer",
            fontFamily: "var(--font-body)",
          }}
        >
          Yes, I'm safe — continue
        </button>
      </div>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────
// SCREEN 4 — SUMMARY
// ─────────────────────────────────────────────────────────────────────

const SummaryScreen = ({ sessionData, onRestart }) => {
  const { messages = [], startTime = Date.now(), finalState = null } = sessionData || {};
  const duration = Math.floor((Date.now() - startTime) / 1000);
  const durationStr = duration < 60
    ? `${duration}s`
    : `${Math.floor(duration / 60)}m ${duration % 60}s`;

  const userTurns = messages.filter(m => m.isUser).length;
  const chroma = CHROMA["neutral"];

  return (
    <div style={{
      position: "relative", zIndex: 1,
      height: "100vh", overflowY: "auto",
      padding: "60px 40px",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <div style={{ maxWidth: 540, width: "100%", animation: "fade-up 0.8s ease" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{
            width: 56, height: 56, borderRadius: "50%",
            background: "radial-gradient(circle at 40% 40%, #4ecdc4, rgba(78,205,196,0.2))",
            margin: "0 auto 20px",
            animation: "breathe 4s ease infinite",
          }} />
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: "2rem", fontWeight: 300,
            marginBottom: 10, color: "var(--text)",
          }}>
            Session complete
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: 15 }}>
            You showed up for yourself today. That matters.
          </p>
        </div>

        {/* Stats */}
        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
          gap: 12, marginBottom: 32,
        }}>
          {[
            { label: "Duration", value: durationStr },
            { label: "Exchanges", value: userTurns },
            { label: "State", value: finalState?.trajectory || "—" },
          ].map(({ label, value }) => (
            <div key={label} style={{
              padding: "16px", textAlign: "center",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
            }}>
              <div style={{ fontSize: 22, fontFamily: "var(--font-mono)", color: "#4ecdc4", marginBottom: 4 }}>
                {value}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                {label}
              </div>
            </div>
          ))}
        </div>

        {/* Emotional arc */}
        {finalState && (
          <div style={{
            padding: 20,
            background: "rgba(78,205,196,0.05)",
            border: "1px solid rgba(78,205,196,0.12)",
            borderRadius: "var(--radius)",
            marginBottom: 24,
          }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 16 }}>
              SESSION EMOTIONAL ARC
            </div>
            <ValenceArousalGauge
              valence={finalState.valence || 0}
              arousal={finalState.arousal || 0}
              chroma={chroma}
            />
            {finalState.trajectory && (
              <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-muted)" }}>
                Trajectory ended as{" "}
                <span style={{
                  color: finalState.trajectory === "improving" ? "#4ecdc4"
                    : finalState.trajectory === "escalating" ? "#e07070"
                      : "var(--text-muted)"
                }}>
                  {finalState.trajectory}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Resources — always shown at end of session */}
        <div style={{
          padding: "16px 20px",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          marginBottom: 32,
          fontSize: 13, color: "var(--text-muted)", lineHeight: 1.7,
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-faint)", marginBottom: 8 }}>
            REMEMBER
          </div>
          EmpathEase is a supportive tool, not a replacement for professional care.
          If you found today's session helpful, speaking with a licensed professional can go even deeper.
          <div style={{ marginTop: 12, fontSize: 12 }}>
            iCall: <span style={{ color: "#c4956a", fontFamily: "var(--font-mono)" }}>9152987821</span>
            {" "}·{" "}
            Vandrevala: <span style={{ color: "#c4956a", fontFamily: "var(--font-mono)" }}>1860-2662-345</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
          <button
            onClick={onRestart}
            style={{
              padding: "12px 28px",
              background: "rgba(78,205,196,0.1)",
              border: "1px solid rgba(78,205,196,0.3)",
              borderRadius: 30,
              color: "#4ecdc4",
              fontSize: 14, cursor: "pointer",
              fontFamily: "var(--font-body)",
            }}
          >
            New session
          </button>
          <button
            onClick={() => window.close()}
            style={{
              padding: "12px 28px",
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: 30,
              color: "var(--text-muted)",
              fontSize: 14, cursor: "pointer",
              fontFamily: "var(--font-body)",
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// APP ROOT
// ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [screen, setScreen] = useState("landing");  // landing|session|crisis|summary
  const [sessionId, setSessionId] = useState(null);
  const [crisisData, setCrisisData] = useState(null);
  const [sessionData, setSessionData] = useState({});
  const [emotion, setEmotion] = useState("neutral");
  const [isCrisis, setIsCrisis] = useState(false);

  const messagesRef = useRef([]);
  const sessionStart = useRef(null);
  const finalStateRef = useRef(null);

  const handleStart = () => {
    const id = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setSessionId(id);
    sessionStart.current = Date.now();
    messagesRef.current = [];
    setScreen("session");
  };

  const handleCrisis = (data) => {
    setCrisisData(data);
    setIsCrisis(true);
    setScreen("crisis");
  };

  const handleCrisisContinue = () => {
    setIsCrisis(false);
    setScreen("session");
  };

  const handleEnd = () => {
    setSessionData({
      messages: messagesRef.current,
      startTime: sessionStart.current,
      finalState: finalStateRef.current,
    });
    setScreen("summary");
  };

  const handleRestart = () => {
    setEmotion("neutral");
    setIsCrisis(false);
    setCrisisData(null);
    setScreen("landing");
  };

  const chroma = getChroma(emotion, isCrisis);

  return (
    <>
      <GlobalStyles />
      <ChromaBackground emotion={emotion} isCrisis={isCrisis} />

      {screen === "landing" && (
        <LandingScreen onStart={handleStart} />
      )}

      {screen === "session" && (
        <SessionScreen
          sessionId={sessionId}
          onCrisis={handleCrisis}
          onEnd={handleEnd}
          onEmotionChange={setEmotion}
        />
      )}

      {screen === "crisis" && (
        <CrisisScreen
          data={crisisData}
          onContinue={handleCrisisContinue}
        />
      )}

      {screen === "summary" && (
        <SummaryScreen
          sessionData={sessionData}
          onRestart={handleRestart}
        />
      )}
    </>
  );
}