using System.Diagnostics;
using System.Collections.Concurrent;

namespace axis_api.services
{
    public class StreamingService
    {
        private ConcurrentDictionary<int, Process> _activeStreams = new ConcurrentDictionary<int, Process>();

        public async Task StartStreaming(int cameraId, string rtspUrl)
        {
            // Ensure only one stream per camera
            if (_activeStreams.ContainsKey(cameraId))
            {
                Console.WriteLine($"Stream for camera {cameraId} is already running.");
                return;
            }

            string directoryPath = $"camera_id{cameraId}";
            Directory.CreateDirectory(directoryPath); // CreateDirectory is ok if already exists

            var ffmpegArgs = $"-i \"{rtspUrl}\" -c copy -f hls -hls_time 2 -hls_segment_filename {directoryPath}/frame%03d.ts -hls_list_size 40 -hls_flags delete_segments {directoryPath}/output.m3u8";
            
            var startInfo = new ProcessStartInfo
            {
                FileName = "/usr/bin/ffmpeg",
                Arguments = ffmpegArgs,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true // To capture error stream
            };
            await Task.Run(() =>
            {
                var process = new Process { StartInfo = startInfo };
                process.EnableRaisingEvents = true;

                process.Exited += (sender, args) =>
                {
                    _activeStreams.TryRemove(cameraId, out _);
                    Console.WriteLine($"Stream for camera {cameraId} exited.");
                    // Consider additional cleanup and logging
                };

                _activeStreams[cameraId] = process;
                process.Start();

                // Optionally, read the output asynchronously
                _ = Task.Run(() => LogOutputAsync(process));
            });
        }
        private async Task LogOutputAsync(Process process)
        {
            string output = await process.StandardOutput.ReadToEndAsync();
            string errors = await process.StandardError.ReadToEndAsync();

            // Log or handle the output and errors
            Console.WriteLine(output);
            Console.WriteLine(errors);
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