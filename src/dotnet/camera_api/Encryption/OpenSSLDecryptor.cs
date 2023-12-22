using System.Diagnostics;
using System.IO;

namespace axis_api.encryption
{
    public class OpenSSLDecryptor
    {
        public static string DecryptString(string encryptedData, string password)
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "/usr/bin/bash",
                    Arguments = $"-c \"openssl enc -aes-256-cbc -d -pass pass:{password} -pbkdf2 -iter 100000 -base64 <<< '{encryptedData}'\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();

            string result = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            return result;
        }
    }
}
