using System.Diagnostics;
using System.Collections.Concurrent;
using axis_api.encryption;
using System.Net;

namespace axis_api.services
{
    public class StreamingService
    {
        private ConcurrentDictionary<int, Process> _activeStreams = new ConcurrentDictionary<int, Process>();

        public async Task StartStreaming(int cameraId, EncryptedData encryptedData)
        {
            // Ensure only one stream per camera
            if (_activeStreams.ContainsKey(cameraId))
            {
                Console.WriteLine($"Stream for camera {cameraId} is already running.");
                return;
            }
            string rtspUrl = $"rtsp://{OpenSSLDecryptor.DecryptString(
                encryptedData.username ?? throw new InvalidOperationException("Invalid username"),
                Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
            )}:{OpenSSLDecryptor.DecryptString(
                encryptedData.password ?? throw new InvalidOperationException("Invalid password"),
                Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
            )}@{OpenSSLDecryptor.DecryptString(
                encryptedData.ip ?? throw new InvalidOperationException("Invalid IP Address"),
                Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
            )}:{OpenSSLDecryptor.DecryptString(
                encryptedData.port ?? throw new InvalidOperationException("Invalid port"),
                Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
            )}{OpenSSLDecryptor.DecryptString(
                encryptedData.endString ?? throw new InvalidOperationException("Invalid end string"),
                Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
            )}";

            if (!IsValidRtspUrl(rtspUrl))
            {
                Console.WriteLine($"Invalid RTSP URL for camera {cameraId}.");
                return; // Exit the method if the URL is invalid
            }

            string directoryPath = $"streams/camera_id{cameraId}";
            Directory.CreateDirectory(directoryPath); // CreateDirectory is ok if already exists

            List<string> ffmpegArgs = new List<string>()
            {
                "-i", rtspUrl, // Input file
                "-c", "copy", // Codec options
                "-f", "hls", // Format
                "-hls_time", "1", // Segment length
                "-hls_segment_filename", $"{directoryPath}/frame%03d.ts", // Segment filename pattern
                "-hls_list_size", "40", // Maximum number of playlist entries
                "-hls_flags", "delete_segments", $"{directoryPath}/output.m3u8" // HLS flags
            };
            var startInfo = new ProcessStartInfo
            {
                FileName = "/usr/bin/ffmpeg",
                Arguments = string.Join(" ", ffmpegArgs), // Join the arguments into a single string
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true
            };

            await Task.Run(() =>
            {
                var process = new Process { StartInfo = startInfo };
                process.EnableRaisingEvents = true;
                process.OutputDataReceived += (sender, args) => Console.WriteLine(args.Data);
                process.ErrorDataReceived += (sender, args) => Console.WriteLine(args.Data);

                process.Exited += (sender, args) =>
                {
                    _activeStreams.TryRemove(cameraId, out _);
                    Console.WriteLine($"Stream for camera {cameraId} exited.");
                };

                _activeStreams[cameraId] = process;
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            });
        }
        public bool IsValidRtspUrl(string rtspUrl)
        {
            // Try to create a Uri object from the RTSP URL.
            if (!Uri.TryCreate(rtspUrl, UriKind.Absolute, out Uri? parsedUri))
            {
                return false; // Not a valid URI
            }

            // Check if the scheme is RTSP
            if (!parsedUri.Scheme.Equals("rtsp", StringComparison.OrdinalIgnoreCase))
            {
                return false; // Not an RTSP URL
            }

            // Validate the IP address part of the URL
            if (!IPAddress.TryParse(parsedUri.Host, out IPAddress? ipAddr))
            {
                return false; // The host part is not a valid IP address
            }

            // Check if the IP address is a local (private) IP address
            if (!IsLocalIpAddress(ipAddr))
            {
                return false; // The IP address is not a local IP address
            }

            // Validate the port
            if (parsedUri.Port != 554)
            {
                return false; // The port is not within the valid range
            }

            return true; // The URL is a valid RTSP URL with a local IP address
        }
        private bool IsLocalIpAddress(IPAddress ipAddress)
        {
            // Check for private IP ranges
            byte[] ipBytes = ipAddress.GetAddressBytes();
            switch (ipBytes[0])
            {
                case 10:
                    return true; // Class A private network (10.x.x.x)
                case 172:
                    return ipBytes[1] >= 16 && ipBytes[1] <= 31; // Class B private networks (172.16.x.x to 172.31.x.x)
                case 192:
                    return ipBytes[1] == 168; // Class C private network (192.168.x.x)
                default:
                    return false; // Not a local IP address
            }
        }
        public void StopStreaming(int cameraId)
        {
            if (_activeStreams.TryRemove(cameraId, out Process? process))
            {
                try
                {
                    if (!process.HasExited)
                    {
                        process.Kill();
                        process.WaitForExit();
                        Console.WriteLine($"Stopped streaming for camera {cameraId}.");
                    }
                    else
                    {
                        Console.WriteLine($"Streaming for camera {cameraId} has already exited.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error stopping streaming for camera {cameraId}: {ex.Message}");
                }
                finally
                {
                    process.Dispose();
                }
            }
            else
            {
                Console.WriteLine($"No active stream found for camera {cameraId}.");
            }
        }
        public void StopAllStreams()
        {
            foreach (var entry in _activeStreams.Keys.ToList())
            {
                if (_activeStreams.TryRemove(entry, out Process? process))
                {
                    if (!process.HasExited)
                    {
                        process.Kill();
                        process.WaitForExit();
                        process.Dispose();
                    }
                }
            }
        }
    }
}