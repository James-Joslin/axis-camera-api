using Microsoft.AspNetCore.Mvc;
using axis_api.services;

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

            int workerThreads, completionPortThreads;
            ThreadPool.GetMaxThreads(out workerThreads, out completionPortThreads);

            Console.WriteLine($"Max worker threads: {workerThreads}, Max completion port threads: {completionPortThreads}");

        }

        [HttpPost("start")]
        public async Task<IActionResult> StartStream(int cameraId)
        {
            if (cameraId <= 0 || cameraId > 4) // Replace MaxCameraId with your actual maximum.
            {
                throw new ArgumentOutOfRangeException("cameraId is out of range.");
            }
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
                        encryptedData
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
        // [HttpPost("stop")]
        // public IActionResult StopStream(int cameraId)
        // {
        //     _streamingService.StopStreaming(cameraId);
        //     return Ok();
        // }
    }
}
