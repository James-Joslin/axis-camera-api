import React, { useState, useRef } from 'react';
import ReactPlayer from 'react-player';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause } from '@fortawesome/free-solid-svg-icons';
import { faCircleXmark } from '@fortawesome/free-regular-svg-icons';

function Modal({ isOpen, close, url }) {
  const [playing, setPlaying] = useState(true); // Set this to true
  const [played, setPlayed] = useState(0);
  const playerRef = useRef(null);

  if (!isOpen) return null;

  const handlePlayPause = () => {
    setPlaying(!playing);
  };

  const handleProgress = (state) => {
    setPlayed(state.played);
  };

  const handleSeekChange = (e) => {
    const newPlayed = parseFloat(e.target.value);
    setPlayed(newPlayed);
    playerRef.current.seekTo(newPlayed);
  };

  return (
    <div className="modal-overlay" onClick={close}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <ReactPlayer
          ref={playerRef}
          url={url}
          playing={playing}
          onProgress={handleProgress}
          width="100%"
          height="100%"
          config={{
            file: {
              attributes: {
                controlsList: 'nodownload noremoteplayback', // This will remove download button and remote playback option
              },
            },
          }}
        />
        
        <div className="custom-controls">
          <button className="play-pause-button" onClick={handlePlayPause}>
            {playing ? (
              <FontAwesomeIcon icon={faPause} style={{ color: "#ffbe6f" }} />
            ) : (
              <FontAwesomeIcon icon={faPlay} style={{ color: "#2ec27e" }} />
            )}
          </button>
          <input
            className="scrubber"
            type="range"
            min={0}
            max={1}
            step="any"
            value={played}
            onChange={handleSeekChange}
          />
        </div>
        <button onClick={close} className="close-modal">
            <FontAwesomeIcon icon={faCircleXmark} style={{ color: "#ffbe6f" }} />
        </button>
      </div>
    </div>
  );
}

export default Modal;
