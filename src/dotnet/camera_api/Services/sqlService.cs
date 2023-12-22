using System;
using Npgsql;
using System.IO;
using System.Threading.Tasks;
using axis_api.encryption;

public struct EncryptedData
{
    public string? ip { get; set; }
    public string? port { get; set; }
    public string? username { get; set; }
    public string? password { get; set; }
    public string? endString { get; set; }
}

public partial class SqlService
{
    private readonly string _encConnectionString;
    private readonly Dictionary<string, string> _sqlPaths;

    public SqlService(IConfiguration configuration)
    {
        _sqlPaths = configuration.GetSection("SqlPaths").Get<Dictionary<string, string>>() ?? throw new InvalidOperationException("No SQL directories found") ;
        if (!_sqlPaths.Any())
        {
            throw new InvalidOperationException("SQL query files not loaded.");
        }
        _encConnectionString = Environment.GetEnvironmentVariable("ENC_CAMERAS_SQLSERVER_CONNECTION_STRING") ?? throw new InvalidOperationException("Invalid connection string");
    }

    public async Task<T> ExecuteSqlQueryAsync<T>(string sqlQuery, Func<NpgsqlDataReader, Task<T>> processResult, Dictionary<string, object>? parameters = null)
    {
        try
        {
            using var connection = new NpgsqlConnection(
                OpenSSLDecryptor.DecryptString(
                    _encConnectionString,
                    Environment.GetEnvironmentVariable("ENC_KEY") ?? throw new InvalidOperationException("Invalid decryption key")
                )
            );
            await connection.OpenAsync();
            using var command = new NpgsqlCommand(sqlQuery, connection);
            
            if (parameters != null)
            {
                foreach (var param in parameters)
                {
                    command.Parameters.AddWithValue(param.Key, param.Value ?? DBNull.Value);
                }
            }

            using var reader = await command.ExecuteReaderAsync();
            return await processResult(reader);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error executing SQL query: {ex.Message}");
        }
    }

    public string GetSqlQueryPath(string key)
    {
        if (_sqlPaths.TryGetValue(key, out var path))
        {
            return path;
        }
        else
        {
            throw new ArgumentException($"SQL path for key '{key}' not found.");
        }
    }

}