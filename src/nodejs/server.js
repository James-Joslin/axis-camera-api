const express = require('express');
const path = require('path');
const axios = require('axios');
const https = require('https');
const { default: helmet } = require('helmet');
const app = express();
const port = 3001;

app.use(helmet())

const rateLimit = require('express-rate-limit');
// const limiter = rateLimit({
//   windowMs: 15 * 60 * 1000, // 15 minutes
//   max: 100 // limit each IP to 100 requests per windowMs
// });
// app.use(limiter);

// app.use(express.json({ limit: '10kb' })); // 10kb limit
// app.use(express.urlencoded({ extended: true, limit: '10kb' }));

// const cors = require('cors');
// const corsOptions = {
//   origin: '', // replace with your domain
//   optionsSuccessStatus: 200,
// };
// app.use(cors(corsOptions));


// Axios instance to accept self-signed certificates in development
const axiosInstance = axios.create({
  httpsAgent: new https.Agent({  
    rejectUnauthorized: false // Accept self-signed certificates
  })
});

// Directory where HLS (.m3u8 and .ts) files are stored.
// This should point to the parent directory of your camera_id folders
const hlsFolderPath = path.join(__dirname, '../dotnet/camera_api/streams/');
app.use('/hls', express.static(hlsFolderPath));

// Basic route for testing that the server is running
app.get('/', (req, res) => res.send('Server is up and running!'));

// Function to add a delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Function to start camera streams by making requests to the C# API
async function startCameraStreams(cameraIds) {
  for (const id of cameraIds) {
    try {
      // Modify this URL to match your C# API's endpoint and ensure it's reachable from this server
      const response = await axiosInstance.post(`https://localhost:5001/Streaming/start?cameraId=${id}`);
      console.log(`Started stream for camera ${id}:`, response.data);

      // Delay before the next request (e.g., 5000 milliseconds = 5 seconds)
      await delay(5000);
    } catch (error) {
      console.error(`Error starting stream for camera ${id}:`, error);
    }
  }
}

// Delay the initial camera stream requests by 5 seconds after server start
setTimeout(() => {
  startCameraStreams([1, 2]); // Add more camera IDs as needed
}, 5000);

// Delay the server start by 5 seconds (5000 milliseconds)
setTimeout(() => {
  app.listen(port, () => console.log(`Server listening on port ${port}`));
}, 5000);
