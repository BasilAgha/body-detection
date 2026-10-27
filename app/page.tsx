
import CameraView from "../src/components/CameraView";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-100 px-4 py-10">
      <h1 className="mb-3 text-3xl font-bold text-gray-900">
        AI Body Shape Detection
      </h1>

      <p className="mb-8 text-gray-600">
        Stand in front of the camera to begin.
      </p>

      <CameraView />
    </main>
  );
}
