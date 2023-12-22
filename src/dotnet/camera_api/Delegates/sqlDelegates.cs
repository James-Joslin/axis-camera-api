using System;
using Npgsql;
using System.IO;
using System.Threading.Tasks;

public delegate Task<EncryptedData> GetCameraData(NpgsqlDataReader reader);
