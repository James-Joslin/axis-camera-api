using Microsoft.AspNetCore.Mvc;
using axis_api.services;
using axis_api.encryption;

namespace axis_api.controllers
{
    [ApiController]
    [Route("[controller]")]
    public class StreamingController : ControllerBase
    {
        private readonly StreamingService _streamingService;
        private readonly SqlService _sqlService;

        public StreamingController(StreamingService streamingService, SqlService sqlService)
        {
            _streamingService = streamingService;
            _sqlService = sqlService;
        }

        [HttpPost("start")]
        public async Task<IActionResult> StartStream(int cameraId)
        {
            try
            {
                string sqlQuery = System.IO.File.ReadAllText(_sqlService.GetSqlQueryPath("GetCameraData"));
                Dictionary<string, object> parameters = new Dictionary<string, object>
                {
                    { "@elementId", cameraId }
                };
                EncryptedData encryptedData = await _sqlService.ExecuteSqlQueryAsync(sqlQuery, SqlService.GetCameraData, parameters);

                // Check if encryptedData is not null and has valid data
                if (!string.IsNullOrEmpty(encryptedData.port) && !string.IsNullOrEmpty(encryptedData.ip))
                {
                    await _streamingService.StartStreaming(
                        cameraId,
                        $"rtsp://{OpenSSLDecryptor.DecryptString(
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
                        )}"
                    );
                    return Ok();
                }
                else
                {
                    // Return a NotFound (or similar) if no data was found
                    return NotFound($"Incomplete data found for camera ID {cameraId}.");
                }
                
            }
            catch (Exception ex)
            {
                // Log the exception (consider using a logging framework)
                Console.WriteLine(ex.Message);

                // Return an error response
                return StatusCode(500, "An error occurred while processing your request: " + ex.Message);
            }
        }
        [HttpPost("stop")]
        public IActionResult StopStream(int cameraId)
        {
            _streamingService.StopStreaming(cameraId);
            return Ok();
        }
    }
}
