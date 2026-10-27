
"use client";

import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const WASM_PATH = "/mediapipe/wasm";

// MediaPipe sometimes logs informational messages as errors.
// Filter only the known TensorFlow Lite INFO message.
function filterMediaPipeInfo() {
  const originalError = console.error;

  console.error = (...args: unknown[]) => {
    const message = args.map(String).join(" ");

    if (
      message.includes(
        "INFO: Created TensorFlow Lite XNNPACK delegate for CPU"
      )
    ) {
      return;
    }

    originalError.apply(console, args);
  };

  return () => {
    console.error = originalError;
  };
}

export default function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);

  const frameRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const startingRef = useRef(false);
  const mountedRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);

  const [cameraActive, setCameraActive] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [error, setError] = useState("");
  const [bodyDetected, setBodyDetected] = useState(false);

  // Load MediaPipe model
  useEffect(() => {
    let cancelled = false;
    let model: PoseLandmarker | null = null;

    mountedRef.current = true;

    const restoreConsole = filterMediaPipeInfo();

    async function loadModel() {
      try {
        const vision =
          await FilesetResolver.forVisionTasks(WASM_PATH);

        if (cancelled) return;

        const createdModel =
          await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: MODEL_URL,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });

        model = createdModel;

        if (cancelled) {
          try {
            createdModel.close();
          } catch {
            // Ignore cleanup failures.
          }
          return;
        }

        poseRef.current = createdModel;
        setModelReady(true);
        setError("");
      } catch (err) {
        if (!cancelled) {
          const message =
            err instanceof Error
              ? err.message
              : String(err);

          console.error("MediaPipe initialization:", message);
          setError(`Model loading failed: ${message}`);
        }
      }
    }

    loadModel();

    return () => {
      cancelled = true;
      mountedRef.current = false;
      runningRef.current = false;

      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }

      streamRef.current
        ?.getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;

      if (model) {
        if (poseRef.current === model) {
          poseRef.current = null;
        }

        try {
          model.close();
        } catch {
          // Ignore cleanup failures.
        }
      }

      restoreConsole();
    };
  }, []);

  // Draw body landmarks
  function detectBody() {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const pose = poseRef.current;

    if (
      video &&
      canvas &&
      pose &&
      video.readyState >= 2
    ) {
      const width = video.videoWidth;
      const height = video.videoHeight;

      if (width > 0 && height > 0) {
        if (canvas.width !== width) {
          canvas.width = width;
        }

        if (canvas.height !== height) {
          canvas.height = height;
        }

        const ctx = canvas.getContext("2d");

        if (
          ctx &&
          video.currentTime !== lastVideoTimeRef.current
        ) {
          lastVideoTimeRef.current = video.currentTime;

          ctx.clearRect(0, 0, width, height);

          try {
            const results = pose.detectForVideo(
              video,
              performance.now()
            );

            const detected =
              results.landmarks.length > 0;

            setBodyDetected((previous) =>
              previous === detected ? previous : detected
            );

            const drawing = new DrawingUtils(ctx);

            for (const landmarks of results.landmarks) {
              drawing.drawConnectors(
                landmarks,
                PoseLandmarker.POSE_CONNECTIONS,
                {
                  color: "#00FF88",
                  lineWidth: 3,
                }
              );

              drawing.drawLandmarks(landmarks, {
                color: "#FF4444",
                radius: 4,
              });
            }
          } catch (err) {
            const message =
              err instanceof Error
                ? err.message
                : String(err);

            console.warn("Detection exception:", message);

            runningRef.current = false;
            setCameraActive(false);
            setBodyDetected(false);
            setError(`Body tracking failed: ${message}`);

            streamRef.current
              ?.getTracks()
              .forEach((track) => track.stop());

            streamRef.current = null;
            video.srcObject = null;
            return;
          }
        }
      }
    }

    frameRef.current =
      requestAnimationFrame(detectBody);
  }

  // Start camera
  async function startCamera() {
    if (
      !modelReady ||
      !poseRef.current ||
      startingRef.current ||
      runningRef.current
    ) {
      return;
    }

    startingRef.current = true;
    setError("");

    let stream: MediaStream | null = null;

    try {
      stream =
        await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const video = videoRef.current;

      if (!video) {
        throw new Error("Video element not found.");
      }

      streamRef.current = stream;
      video.srcObject = stream;

      await video.play();

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      lastVideoTimeRef.current = -1;
      runningRef.current = true;

      setCameraActive(true);
      setBodyDetected(false);

      frameRef.current =
        requestAnimationFrame(detectBody);
    } catch (err) {
      stream?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (mountedRef.current) {
        const message =
          err instanceof Error
            ? err.message
            : String(err);

        setError(`Camera failed: ${message}`);
        setCameraActive(false);
      }
    } finally {
      startingRef.current = false;
    }
  }

  // Stop camera
  function stopCamera() {
    runningRef.current = false;

    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }

    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;

    if (canvas) {
      canvas
        .getContext("2d")
        ?.clearRect(0, 0, canvas.width, canvas.height);
    }

    lastVideoTimeRef.current = -1;

    setCameraActive(false);
    setBodyDetected(false);
  }

  return (
    <div className="flex flex-col items-center gap-5">
      <div className="relative aspect-[3/4] w-full max-w-md overflow-hidden rounded-2xl bg-slate-900">

        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 h-full w-full -scale-x-100 object-contain"
        />

        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-contain"
        />

        {!cameraActive && (
          <div className="absolute inset-0 flex items-center justify-center text-white">
            {modelReady
              ? "Camera is off"
              : error
              ? "Model unavailable"
              : "Loading AI model..."}
          </div>
        )}

        {cameraActive && (
          <div className="absolute left-3 top-3 rounded-lg bg-black/70 px-3 py-2 text-sm text-white">
            {bodyDetected
              ? "Body Detected"
              : "Searching for body..."}
          </div>
        )}
      </div>

      {error && (
        <p className="max-w-md text-center text-sm text-red-500">
          {error}
        </p>
      )}

      <button
        onClick={cameraActive ? stopCamera : startCamera}
        disabled={!modelReady}
        className="rounded-xl bg-blue-600 px-8 py-3 font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {cameraActive
          ? "Stop Camera"
          : modelReady
          ? "Start Camera"
          : "Loading Model..."}
      </button>

      <p className="text-center text-sm text-gray-500">
        Stand facing the camera with your full body visible.
      </p>
    </div>
  );
}
