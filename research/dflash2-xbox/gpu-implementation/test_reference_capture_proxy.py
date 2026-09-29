"""Offline regression for trusted CPU rerecord; no console/network access."""
import argparse
import io
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import reference_capture_proxy as proxy


def main():
    source = Path(__file__).parent / "f0-2026-09-28/reference.cap"
    raw = source.read_bytes()

    class Executor:
        def __init__(self, bad_status=False):
            self.source = io.BytesIO(raw[8:])
            self.responses = bytearray()
            self.count = 0
            self.bad_status = bad_status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, seconds):
            pass

        def sendall(self, request):
            n, m = proxy.RECORD.unpack(self.source.read(8))
            assert request == self.source.read(n)
            response = bytearray(self.source.read(m))
            if self.bad_status:
                struct.pack_into("<i", response, 16, -1)
            self.responses.extend(response)
            self.count += 1

        def recv(self, size):
            # Partial TCP reads must also work.
            chunk = bytes(self.responses[:min(size, 113)])
            del self.responses[:len(chunk)]
            return chunk

    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "reference.cap"
        args = argparse.Namespace(input=source, output=output, target="localhost:1")
        executor = Executor()
        with patch.object(proxy.socket, "create_connection", return_value=executor):
            proxy.rerecord(args)
        assert executor.count == 8
        assert output.read_bytes() == raw[:output.stat().st_size]
        output.unlink()
        with patch.object(proxy.socket, "create_connection", return_value=Executor(True)):
            try:
                proxy.rerecord(args)
            except ValueError as error:
                assert "invalid CPU response" in str(error)
            else:
                raise AssertionError("failed CPU status accepted")
        assert not output.exists()
        corrupt = Path(temp) / "corrupt.cap"
        corrupt.write_bytes(raw[:128])
        args.input = corrupt
        with patch.object(proxy.socket, "create_connection") as connect:
            try:
                proxy.rerecord(args)
            except ValueError as error:
                assert "trusted reference" in str(error)
            else:
                raise AssertionError("corrupt source accepted")
            connect.assert_not_called()
    print("rerecord: 8 frames, partial reads, failed response, corrupt source PASS")


if __name__ == "__main__":
    main()
