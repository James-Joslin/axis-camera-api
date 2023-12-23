import React, { useState } from 'react';
import ReactPlayer from 'react-player';
import './App.css'; // Make sure the path is correct

function App() {
  const [cameraIds] = useState([1, 2]); // Initialize with your camera IDs

  return (
    <div className="stream-grid">
      {cameraIds.map(cameraId => (
        <div key={cameraId} className="stream-panel">
          <h2>Camera {cameraId}</h2>
          <ReactPlayer
            url={`/hls/camera_id${cameraId}/output.m3u8`}
            playing
            controls
            muted // This mutes the video by default
            onError={(e) => console.error('Error playing video:', e)}
            onReady={() => console.log(`Player for camera ${cameraId} is ready`)}
          />
        </div>
      ))}
    </div>
  );
}

export default App;

