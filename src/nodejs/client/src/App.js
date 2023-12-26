import React, { useState } from 'react';
import ReactPlayer from 'react-player';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause } from '@fortawesome/free-solid-svg-icons';
import Modal from './Modal';
import './App.css';

function App() {
  const [cameraIds] = useState([1, 2, 3]);
  const [playing, setPlaying] = useState(cameraIds.reduce((acc, id) => ({ ...acc, [id]: true }), {}));

  const handlePlayPause = (cameraId) => {
    setPlaying(prevPlaying => ({
      ...prevPlaying,
      [cameraId]: !prevPlaying[cameraId]
    }));
  };

  const [modalOpen, setModalOpen] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState(null);

  const handleExpandClick = (cameraId) => {
    setPlaying(prevPlaying => ({ 
      ...prevPlaying, 
      [cameraId]: false // Stop the playback of the clicked camera
    }));
    setSelectedCamera(cameraId);
    setModalOpen(true);
  };
  
  const closeModal = () => {
    if (selectedCamera !== null) {
      setPlaying(prevPlaying => ({ 
        ...prevPlaying, 
        [selectedCamera]: true // Resume playback when modal closes
      }));
    }
    setModalOpen(false);
  };


  return (
    <>
      <header className="header">
        Home Surveillance
      </header>
      <div className="stream-grid">
        {cameraIds.map(cameraId => (
          <div key={cameraId} className="stream-panel">
            <div className="camera-label">Camera {cameraId}</div>
            <div className="react-player-wrapper">
              {playing[cameraId] ? (
                <ReactPlayer
                  className="react-player"
                  url={`/streams/camera_id${cameraId}/playlist.m3u8`}
                  playing={playing[cameraId]}
                  muted
                  width="100%"
                  height="100%"
                  onError={(e) => console.error('Error playing video:', e)}
                  onReady={() => console.log(`Player for camera ${cameraId} is ready`)}
                />
              ) : (
                <div className="black-screen"> {/* or a placeholder image */}
                  {/* Optionally, you can display an icon or message here */}
                </div>
              )}
            </div>
            <button className="play-pause-button" onClick={() => handlePlayPause(cameraId)}>
              {playing[cameraId] ? (
                <FontAwesomeIcon icon={faPause} style={{ color: "#ffbe6f" }} />
              ) : (
                <FontAwesomeIcon icon={faPlay} style={{ color: "#2ec27e" }} />
              )}
            </button>
            <button className="expand-button" onClick={() => handleExpandClick(cameraId)}>
              Expand
            </button>
          </div>
        ))}
      </div>
      <Modal
        className = "modal-player"
        isOpen={modalOpen}
        close={closeModal}
        url={selectedCamera ? `/streams/camera_id${selectedCamera}/playlist.m3u8` : ''}
      />
    </>
  );
}

export default App;
