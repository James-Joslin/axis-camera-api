using System;
using Npgsql;
using System.Threading.Tasks;

public partial class SqlService
{
    public static async Task<EncryptedData> GetCameraData(NpgsqlDataReader reader)
    {
        if (await reader.ReadAsync())
        {
            return new EncryptedData
            {
                ip = reader["ip"].ToString(),
                port = reader["port"].ToString(),
                username = reader["username"].ToString(),
                password = reader["pass"].ToString(),
                endString = reader["end_string"].ToString()
            };
        }
        return default;
    }
    // Other processing methods as needed
}

