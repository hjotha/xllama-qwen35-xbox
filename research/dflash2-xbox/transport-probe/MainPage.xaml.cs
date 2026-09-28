using System;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Windows.Networking.Sockets;
using Windows.Storage;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;

namespace DFlashRpcProbe
{
    public sealed partial class MainPage : Page
    {
        private const int Port = 43125;
        private const int MaxPayload = 1024 * 1024;
        private StreamSocketListener listener;

        public MainPage()
        {
            InitializeComponent();
            Loaded += OnLoaded;
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            if (listener != null) return;
            try
            {
                listener = new StreamSocketListener();
                listener.ConnectionReceived += OnConnection;
                await listener.BindServiceNameAsync(Port.ToString());
                StatusValue.Text = "Listening on TCP " + Port;
            }
            catch (Exception ex)
            {
                listener?.Dispose();
                listener = null;
                StatusValue.Text = "Listener failed: " + ex.Message;
            }
        }

        private async void OnConnection(StreamSocketListener sender,
                                        StreamSocketListenerConnectionReceivedEventArgs args)
        {
            using (var socket = args.Socket)
            {
                try
                {
                    // Accepted StreamSockets default to NoDelay=true. WinRT rejects
                    // changing StreamSocket.Control after the connection is accepted.
                    using (var input = socket.InputStream.AsStreamForRead())
                    using (var output = socket.OutputStream.AsStreamForWrite())
                    using (var sha = SHA256.Create())
                    {
                        var header = new byte[40];
                        var payload = new byte[MaxPayload];
                        var responsePayload = new byte[MaxPayload];
                        while (await ReadExact(input, header, header.Length))
                        {
                            if (header[0] != 'D' || header[1] != 'F' ||
                                header[2] != 'R' || header[3] != '2' ||
                                BitConverter.ToUInt16(header, 4) != 1)
                                break;
                            ushort command = BitConverter.ToUInt16(header, 6);
                            uint length = BitConverter.ToUInt32(header, 32);
                            uint responseLength = command == 3 ? BitConverter.ToUInt32(header, 36) : 0;
                            if (length > MaxPayload || responseLength > MaxPayload) break;

                            long readStart = Stopwatch.GetTimestamp();
                            if (!await ReadExact(input, payload, (int)length)) break;
                            long readEnd = Stopwatch.GetTimestamp();
                            byte[] digest = new byte[32];
                            long hashStart = Stopwatch.GetTimestamp();
                            if (command == 2 || command == 3)
                                digest = sha.ComputeHash(payload, 0, (int)length);
                            long hashEnd = Stopwatch.GetTimestamp();

                            var response = new byte[72];
                            response[0] = (byte)'D'; response[1] = (byte)'F';
                            response[2] = (byte)'R'; response[3] = (byte)'2';
                            Put(response, 4, (ushort)1);
                            Put(response, 6, (ushort)((command == 1 || command == 2 || command == 3) ? 0 : 1));
                            Buffer.BlockCopy(header, 8, response, 8, 8);
                            Put(response, 16, ToMicros(readEnd - readStart));
                            Put(response, 24, ToMicros(hashEnd - hashStart));
                            Put(response, 32, length);
                            Put(response, 36, responseLength);
                            Buffer.BlockCopy(digest, 0, response, 40, 32);
                            await output.WriteAsync(response, 0, response.Length);
                            if (responseLength != 0)
                                await output.WriteAsync(responsePayload, 0, (int)responseLength);
                            await output.FlushAsync();
                        }
                    }
                }
                catch (Exception ex)
                {
                    var file = await ApplicationData.Current.LocalFolder.CreateFileAsync(
                        "rpc-error.txt", CreationCollisionOption.ReplaceExisting);
                    await FileIO.WriteTextAsync(file, ex.ToString());
                }
            }
        }

        private static async Task<bool> ReadExact(Stream stream, byte[] data, int count)
        {
            int offset = 0;
            while (offset < count)
            {
                int read = await stream.ReadAsync(data, offset, count - offset);
                if (read == 0) return false;
                offset += read;
            }
            return true;
        }

        private static ulong ToMicros(long ticks) =>
            (ulong)(ticks * 1000000.0 / Stopwatch.Frequency);

        private static void Put(byte[] destination, int offset, ushort value) =>
            Buffer.BlockCopy(BitConverter.GetBytes(value), 0, destination, offset, 2);
        private static void Put(byte[] destination, int offset, uint value) =>
            Buffer.BlockCopy(BitConverter.GetBytes(value), 0, destination, offset, 4);
        private static void Put(byte[] destination, int offset, ulong value) =>
            Buffer.BlockCopy(BitConverter.GetBytes(value), 0, destination, offset, 8);
    }
}
