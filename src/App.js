import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as faceapi from 'face-api.js';
import './App.css';

// Fallback thresholds used before calibration
const FALLBACK_EAR  = 0.25;
const FALLBACK_MAR  = 0.55;
const NOD_DELTA     = 0.08; // added to calibrated baseline pitch

const EYE_FRAMES  = 15; // ~1.5s
const YAWN_FRAMES = 10; // ~1.0s
const NOD_FRAMES  = 8;  // ~0.8s
const CALIB_MS    = 10000;

// ── Math helpers ──────────────────────────────────────────────────────────────
function arrMean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function arrStd(a)  {
  const m = arrMean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
}

// ── Feature extractors ────────────────────────────────────────────────────────
function getEAR(eye) {
  const A = Math.hypot(eye[1].x - eye[5].x, eye[1].y - eye[5].y);
  const B = Math.hypot(eye[2].x - eye[4].x, eye[2].y - eye[4].y);
  const C = Math.hypot(eye[0].x - eye[3].x, eye[0].y - eye[3].y);
  return (A + B) / (2.0 * C);
}

function getMAR(mouth) {
  // mouth[0]=left corner, mouth[6]=right corner, mouth[3]=top, mouth[9]=bottom
  const v = Math.hypot(mouth[3].x - mouth[9].x, mouth[3].y - mouth[9].y);
  const h = Math.hypot(mouth[0].x - mouth[6].x, mouth[0].y - mouth[6].y);
  return v / h;
}

function getHeadPitch(lm) {
  // Normalized nose-tip Y between eye-center and chin.
  // Increases when head tilts forward (nod).
  const nose      = lm.getNose();
  const jaw       = lm.getJawOutline();
  const eyePts    = [...lm.getLeftEye(), ...lm.getRightEye()];
  const eyeCenterY = eyePts.reduce((s, p) => s + p.y, 0) / eyePts.length;
  const chinY      = jaw[8].y;
  const faceH      = Math.max(chinY - eyeCenterY, 1);
  return (nose[6].y - eyeCenterY) / faceH; // nose[6] = landmark 33, nose tip
}

// ── Canvas drawing ────────────────────────────────────────────────────────────
function drawRegion(ctx, points, color, fillAlpha) {
  if (!points?.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;
  if (fillAlpha > 0) {
    ctx.globalAlpha = fillAlpha;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function drawNodLine(ctx, lm, alerting) {
  const nose = lm.getNose();
  const jaw  = lm.getJawOutline();
  const tip  = nose[6];
  const chin = jaw[8];
  const color = alerting ? '#ef4444' : '#f59e0b';

  ctx.beginPath();
  ctx.moveTo(tip.x, tip.y);
  ctx.lineTo(chin.x, chin.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.setLineDash([5, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;

  for (const p of [tip, chin]) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

// ── Audio ─────────────────────────────────────────────────────────────────────
function createSiren(ctx) {
  const osc  = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = 'sawtooth';
  gain.gain.value = 0.4;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  const sweep = () => {
    const t = ctx.currentTime;
    osc.frequency.cancelScheduledValues(t);
    osc.frequency.setValueAtTime(440, t);
    osc.frequency.linearRampToValueAtTime(880, t + 0.5);
    osc.frequency.linearRampToValueAtTime(440, t + 1.0);
  };
  sweep();
  const id = setInterval(sweep, 1000);
  return { osc, gain, id };
}

function stopSiren(s) {
  if (!s) return;
  clearInterval(s.id);
  try { s.osc.stop(); } catch {}
  s.osc.disconnect();
  s.gain.disconnect();
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h, m, s].map(v => String(v).padStart(2, '0')).join(':');
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const intervalRef  = useRef(null);
  const sirenRef     = useRef(null);
  const audioCtx     = useRef(null);
  const countdownRef = useRef(null);

  // Frame counters
  const eyeCount  = useRef(0);
  const yawnCount = useRef(0);
  const nodCount  = useRef(0);

  // Calibration
  const calibFlag   = useRef(false);
  const calibStart  = useRef(null);
  const earSamples  = useRef([]);
  const marSamples  = useRef([]);
  const pitchSamples = useRef([]);

  // Live thresholds (updated after calibration)
  const earThresh   = useRef(FALLBACK_EAR);
  const marThresh   = useRef(FALLBACK_MAR);
  const pitchThresh = useRef(null);

  // Break alarm — refs so interval closure always reads latest values
  const sessionStartRef  = useRef(null);
  const timerRef         = useRef(null);
  const breakFiredRef    = useRef(false);
  const snoozeUntilRef   = useRef(null);
  const breakEnabledRef  = useRef(true);
  const breakDurationRef = useRef(120);

  // UI state
  const [phase,          setPhase]          = useState('loading');
  const [countdown,      setCountdown]      = useState(3);
  const [calibProgress,  setCalibProgress]  = useState(0);
  const [thresholds,     setThresholds]     = useState(null);
  const [eyeAlert,       setEyeAlert]       = useState(false);
  const [mouthAlert,     setMouthAlert]     = useState(false);
  const [nodAlert,       setNodAlert]       = useState(false);
  const [ear,            setEar]            = useState(null);
  const [mar,            setMar]            = useState(null);
  // Break alarm state
  const [breakEnabled,   setBreakEnabled]   = useState(true);
  const [breakDuration,  setBreakDuration]  = useState(120); // minutes
  const [breakMessage,   setBreakMessage]   = useState('Time to rest! Pull over safely.');
  const [showBreakModal, setShowBreakModal] = useState(false);
  const [sessionElapsed, setSessionElapsed] = useState(0);  // seconds

  const isAlarming = eyeAlert || mouthAlert || nodAlert;

  // Load models
  useEffect(() => {
    const MODEL_URL = process.env.PUBLIC_URL + '/models';
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    ])
      .then(() => setPhase('ready'))
      .catch(() => setPhase('error'));
  }, []);

  // Sync break settings state → refs so interval closure always reads fresh values
  useEffect(() => { breakEnabledRef.current  = breakEnabled;  }, [breakEnabled]);
  useEffect(() => { breakDurationRef.current = breakDuration; }, [breakDuration]);

  // Session timer + break check (1-second tick, only while running)
  useEffect(() => {
    if (phase !== 'running') { clearInterval(timerRef.current); return; }
    timerRef.current = setInterval(() => {
      if (!sessionStartRef.current) return;
      const elapsed = Math.floor((Date.now() - sessionStartRef.current) / 1000);
      setSessionElapsed(elapsed);
      if (breakEnabledRef.current && !breakFiredRef.current) {
        const snoozeOk = !snoozeUntilRef.current || Date.now() >= snoozeUntilRef.current;
        if (snoozeOk && elapsed >= breakDurationRef.current * 60) {
          breakFiredRef.current = true;
          setShowBreakModal(true);
        }
      }
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [phase]);

  // Cleanup on unmount
  useEffect(() => () => {
    clearInterval(intervalRef.current);
    clearInterval(timerRef.current);
    stopSiren(sirenRef.current);
  }, []);

  const startCamera = useCallback(async () => {
    setPhase('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      audioCtx.current = new (window.AudioContext || window.webkitAudioContext)();
    } catch {
      setPhase('error');
    }
  }, []);

  const handleVideoPlay = useCallback(() => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    faceapi.matchDimensions(canvas, { width: video.videoWidth, height: video.videoHeight });

    // 3-2-1 countdown, then start calibration
    setPhase('countdown');
    setCountdown(3);

    let n = 3;
    const tick = () => {
      n -= 1;
      setCountdown(n);
      if (n > 0) { countdownRef.current = setTimeout(tick, 1000); return; }

      // n === 0 — begin calibration
      calibFlag.current    = true;
      calibStart.current   = Date.now();
      earSamples.current   = [];
      marSamples.current   = [];
      pitchSamples.current = [];
      setCalibProgress(0);
      setPhase('calibrating');

      intervalRef.current = setInterval(async () => {
      const det = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks();

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!det) {
        if (!calibFlag.current) {
          // Face lost during monitoring — reset counts and stop alarm
          eyeCount.current = yawnCount.current = nodCount.current = 0;
          setEyeAlert(false); setMouthAlert(false); setNodAlert(false);
          setEar(null); setMar(null);
          stopSiren(sirenRef.current); sirenRef.current = null;
        }
        return;
      }

      const resized = faceapi.resizeResults(det, {
        width: video.videoWidth, height: video.videoHeight,
      });
      const lm       = resized.landmarks;
      const leftEye  = lm.getLeftEye();
      const rightEye = lm.getRightEye();
      const mouth    = lm.getMouth();

      const avgEAR = (getEAR(leftEye) + getEAR(rightEye)) / 2;
      const curMAR = getMAR(mouth);
      const pitch  = getHeadPitch(lm);

      // ── CALIBRATION ──────────────────────────────────────────────────────
      if (calibFlag.current) {
        earSamples.current.push(avgEAR);
        marSamples.current.push(curMAR);
        pitchSamples.current.push(pitch);

        const elapsed  = Date.now() - calibStart.current;
        setCalibProgress(Math.min(Math.round(elapsed / CALIB_MS * 100), 100));
        setEar(avgEAR.toFixed(3));
        setMar(curMAR.toFixed(3));

        // Draw all three regions in neutral indigo during calibration
        drawRegion(ctx, leftEye,            '#6366f1', 0.1);
        drawRegion(ctx, rightEye,           '#6366f1', 0.1);
        drawRegion(ctx, mouth.slice(0, 12), '#6366f1', 0.1);
        drawNodLine(ctx, lm, false);

        if (elapsed >= CALIB_MS && earSamples.current.length >= 20) {
          const em = arrMean(earSamples.current), es = arrStd(earSamples.current);
          const mm = arrMean(marSamples.current), ms = arrStd(marSamples.current);
          const pm = arrMean(pitchSamples.current);

          earThresh.current   = Math.max(em - 1.5 * es, 0.18);
          marThresh.current   = Math.min(mm + 1.5 * ms, 0.80);
          pitchThresh.current = pm + NOD_DELTA;

          const t = {
            ear:   earThresh.current.toFixed(3),
            mar:   marThresh.current.toFixed(3),
            pitch: pitchThresh.current.toFixed(3),
          };
          setThresholds(t);
          calibFlag.current     = false;
          sessionStartRef.current = Date.now();
          breakFiredRef.current = false;
          snoozeUntilRef.current = null;
          setPhase('running');
        }
        return;
      }

      // ── DETECTION ────────────────────────────────────────────────────────
      setEar(avgEAR.toFixed(3));
      setMar(curMAR.toFixed(3));

      // Eye closure (EAR)
      if (avgEAR < earThresh.current) eyeCount.current++;
      else eyeCount.current = 0;
      const eyeFired = eyeCount.current >= EYE_FRAMES;
      setEyeAlert(eyeFired);

      // Yawn (MAR)
      if (curMAR > marThresh.current) yawnCount.current++;
      else yawnCount.current = 0;
      const yawnFired = yawnCount.current >= YAWN_FRAMES;
      setMouthAlert(yawnFired);

      // Head nod (pitch)
      let nodFired = false;
      if (pitchThresh.current !== null) {
        if (pitch > pitchThresh.current) nodCount.current++;
        else nodCount.current = 0;
        nodFired = nodCount.current >= NOD_FRAMES;
        setNodAlert(nodFired);
      }

      // Siren — managed directly, no state→effect indirection
      const alarm = eyeFired || yawnFired || nodFired;
      if (alarm && !sirenRef.current && audioCtx.current) {
        sirenRef.current = createSiren(audioCtx.current);
      } else if (!alarm && sirenRef.current) {
        stopSiren(sirenRef.current);
        sirenRef.current = null;
      }

      // Draw eyes + mouth + head-nod line
      drawRegion(ctx, leftEye,            eyeFired  ? '#ef4444' : '#22c55e', eyeFired  ? 0.35 : 0.08);
      drawRegion(ctx, rightEye,           eyeFired  ? '#ef4444' : '#22c55e', eyeFired  ? 0.35 : 0.08);
      drawRegion(ctx, mouth.slice(0, 12), yawnFired ? '#ef4444' : '#818cf8', yawnFired ? 0.35 : 0.08);
      drawNodLine(ctx, lm, nodFired);
      }, 100);   // end setInterval
    };            // end tick
    countdownRef.current = setTimeout(tick, 1000);
  }, []);

  const stopCamera = useCallback(() => {
    clearTimeout(countdownRef.current);
    clearInterval(intervalRef.current);
    stopSiren(sirenRef.current);
    sirenRef.current = null;

    // Stop all camera tracks
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;

    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    // Reset all counters and thresholds
    eyeCount.current = yawnCount.current = nodCount.current = 0;
    calibFlag.current = false;
    earThresh.current = FALLBACK_EAR;
    marThresh.current = FALLBACK_MAR;
    pitchThresh.current = null;

    setEyeAlert(false); setMouthAlert(false); setNodAlert(false);
    setEar(null); setMar(null);
    setThresholds(null);
    setCalibProgress(0);
    clearInterval(timerRef.current);
    sessionStartRef.current = null;
    breakFiredRef.current   = false;
    snoozeUntilRef.current  = null;
    setSessionElapsed(0);
    setShowBreakModal(false);
    setPhase('ready');
  }, []);

  const handleSnooze = () => {
    snoozeUntilRef.current = Date.now() + 15 * 60 * 1000;
    breakFiredRef.current  = false;
    setShowBreakModal(false);
  };

  const handleDismiss = () => setShowBreakModal(false);

  const alertMessage = () => {
    const parts = [];
    if (eyeAlert)   parts.push('EYES CLOSING');
    if (mouthAlert) parts.push('YAWNING');
    if (nodAlert)   parts.push('HEAD NODDING');
    return parts.join(' + ') + ' — WAKE UP!';
  };

  const showVideo = ['starting', 'countdown', 'calibrating', 'running'].includes(phase);

  return (
    <div className={`app ${isAlarming ? 'alarming' : ''}`}>
      <header className="app-header">
        <div className="header-logo">👁</div>
        <div className="header-text">
          <h1>Drowsy Guard</h1>
          <p className="subtitle">Real-time drowsiness detection</p>
        </div>
      </header>

      <main className="app-main">
        {phase === 'loading' && (
          <div className="status-card">
            <div className="spinner" />
            <p>Loading face detection models...</p>
          </div>
        )}
        {phase === 'error' && (
          <div className="status-card error">
            <p>Failed to load models or access camera.</p>
            <p>Make sure your browser allows camera access.</p>
          </div>
        )}
        {phase === 'ready' && (
          <div className="status-card wide">
            <p>A 10-second calibration runs first to tune detection to your face.</p>
            <button className="start-btn" onClick={startCamera}>Start Camera</button>

            <div className="settings-divider"><span>Break Reminder</span></div>

            <div className="settings-row">
              <span className="settings-label">Enable</span>
              <label className="toggle">
                <input type="checkbox" checked={breakEnabled} onChange={e => setBreakEnabled(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>

            {breakEnabled && (
              <>
                <div className="preset-btns">
                  {[30, 60, 90, 120, 180].map(m => (
                    <button
                      key={m}
                      className={`preset-btn ${breakDuration === m ? 'active' : ''}`}
                      onClick={() => setBreakDuration(m)}
                    >
                      {m < 60 ? `${m}m` : m === 60 ? '1h' : m === 90 ? '1.5h' : m === 120 ? '2h' : '3h'}
                    </button>
                  ))}
                </div>
                <div className="custom-time-row">
                  <span>Custom</span>
                  <input
                    type="number" min="5" max="480"
                    value={breakDuration}
                    onChange={e => setBreakDuration(Math.max(5, Math.min(480, Number(e.target.value))))}
                    className="time-input"
                  />
                  <span>min</span>
                </div>
                <input
                  type="text"
                  className="message-input"
                  value={breakMessage}
                  onChange={e => setBreakMessage(e.target.value)}
                  placeholder="Reminder message..."
                  maxLength={100}
                />
              </>
            )}
          </div>
        )}

        <div className={`video-wrapper ${showVideo ? '' : 'hidden'}`}>

          {phase === 'calibrating' && (
            <div className="calib-card">
              <p className="calib-title">Calibrating — look straight ahead and stay still</p>
              <div className="calib-track">
                <div className="calib-fill" style={{ width: `${calibProgress}%` }} />
              </div>
              <p className="calib-pct">{calibProgress}%</p>
            </div>
          )}

          {isAlarming && (
            <div className="alert-banner">{alertMessage()}</div>
          )}

          <div className="video-container">
            <video ref={videoRef} autoPlay muted playsInline onPlay={handleVideoPlay} />
            <canvas ref={canvasRef} />
            {phase === 'countdown' && (
              <div className="countdown-overlay">
                <span className="countdown-number" key={countdown}>{countdown}</span>
                <p>Get ready — look straight ahead</p>
              </div>
            )}
            <button className="close-btn" onClick={stopCamera} title="Stop camera">✕</button>
          </div>

          {phase === 'running' && sessionElapsed > 0 && (
            <div className="session-timer">
              <div className="timer-item">
                <span className="timer-label">Session</span>
                <span className="timer-value">{formatTime(sessionElapsed)}</span>
              </div>
              {breakEnabled && (
                <div className="timer-item">
                  <span className="timer-label">Break in</span>
                  <span className={`timer-value ${breakDuration * 60 - sessionElapsed < 300 ? 'timer-warn' : ''}`}>
                    {formatTime(Math.max(0, breakDuration * 60 - sessionElapsed))}
                  </span>
                </div>
              )}
            </div>
          )}

          {phase === 'running' && ear !== null && (
            <div className="metrics">
              <div className={`metric ${eyeAlert ? 'bad' : 'good'}`}>
                <span className="metric-label">EAR — Eyes</span>
                <span className="metric-value">{ear}</span>
                <span className="metric-thresh">thresh {thresholds?.ear}</span>
                <span className="metric-status">{eyeAlert ? 'Closing' : 'Open'}</span>
              </div>
              <div className={`metric ${mouthAlert ? 'bad' : 'good'}`}>
                <span className="metric-label">MAR — Mouth</span>
                <span className="metric-value">{mar}</span>
                <span className="metric-thresh">thresh {thresholds?.mar}</span>
                <span className="metric-status">{mouthAlert ? 'Yawning' : 'Closed'}</span>
              </div>
              <div className={`metric ${nodAlert ? 'bad' : 'good'}`}>
                <span className="metric-label">NOD — Head</span>
                <span className="metric-value">{thresholds?.pitch}</span>
                <span className="metric-thresh">calibrated</span>
                <span className="metric-status">{nodAlert ? 'Nodding' : 'Upright'}</span>
              </div>
            </div>
          )}
        </div>
      </main>

      {showBreakModal && (
        <div className="break-backdrop">
          <div className="break-modal">
            <div className="break-icon">REST</div>
            <h2 className="break-title">Break Time!</h2>
            <p className="break-message">{breakMessage}</p>
            <p className="break-session">Driving for {formatTime(sessionElapsed)}</p>
            <div className="break-actions">
              <button className="break-snooze" onClick={handleSnooze}>Snooze 15 min</button>
              <button className="break-dismiss" onClick={handleDismiss}>Taking a Break</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
