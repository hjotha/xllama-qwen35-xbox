"""Exercise real Xbox KV equivalence between v2 controls and fused SYNC."""
import argparse
import hashlib
import json
import socket
import struct

MAGIC = 0x324C4644
REQ = struct.Struct('<IHHQII')
RSP = struct.Struct('<IHHQiIQQQQ')


class Client:
    def __init__(self, host):
        self.sock = socket.create_connection((host, 50053), timeout=30)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.serial = 0
        shape = struct.unpack('<IIIIiIIQ', self.call(1))
        self.enc, self.dec = shape[1:3]
        self.call(6, 0)  # real compute, mock explicitly disabled
        self.call(7)  # no-op capability probe

    def read(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError('connection closed')
            data.extend(chunk)
        return bytes(data)

    def call(self, op, n=0, payload=b'', status=0):
        self.serial += 1
        self.sock.sendall(REQ.pack(MAGIC, 2, op, self.serial, n, len(payload)) + payload)
        rsp = RSP.unpack(self.read(RSP.size))
        result = self.read(rsp[5])
        assert rsp[:4] == (MAGIC, 2, op, self.serial), rsp
        assert rsp[4] == status, (op, rsp)
        return result

    def rows(self, positions):
        return b''.join(struct.pack('<i', p) + struct.pack('<f', (p % 7 - 3) / 100) * self.enc
                        for p in positions)

    def sync(self, positions, cut=None, reset=False):
        payload = self.rows(positions)
        if cut is not None:
            payload = struct.pack('<iI', cut, reset) + payload
        self.call(7 if cut is not None else 3, len(positions), payload)

    def draft(self, pos, n=4):
        data = struct.pack('<iii', pos, 1, n) + b'\0' * ((n + 1) * self.dec * 4)
        result = self.call(4, n + 1, data)
        assert len(result) == (n + 1) * self.dec * 4
        return result

    def setup(self):
        self.call(2)
        self.sync(list(range(8)))
        self.draft(8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='192.168.1.26')
    args = parser.parse_args()
    c = Client(args.host)
    results = []
    try:
        for first in (5, 8, 10, 0):
            c.setup()
            c.call(5, payload=struct.pack('<i', 8))
            if first == 0:
                c.call(2)
            elif first <= 7:
                c.call(5, payload=struct.pack('<i', first))
            positions = list(range(first, first + 4))
            c.sync(positions)
            expected = c.draft(first + 4)
            c.setup()
            c.sync(positions, cut=min(8, first), reset=first == 0)
            actual = c.draft(first + 4)
            assert expected == actual, f'KV output differs at first_pos={first}'
            results.append({'case': f'first_pos_{first}', 'bitwise_equal': True,
                            'sha256': hashlib.sha256(actual).hexdigest()})
        c.call(2)
        c.sync(list(range(64)))
        expected = c.draft(64)
        c.setup()
        c.sync(list(range(64)), cut=0, reset=True)
        assert c.draft(64) == expected
        results.append({'case': 'max_64_rows_reset', 'bitwise_equal': True})
        c.setup()
        expected = c.draft(12)
        c.setup()
        for payload, n in [(struct.pack('<iI', -1, 0) + c.rows([8]), 1),
                           (struct.pack('<iI', 8, 2) + c.rows([8]), 1),
                           (struct.pack('<iI', 8, 1) + c.rows([8]), 1),
                           (struct.pack('<iI', 0, 0), 1),
                           (b'\0', 0)]:
            c.call(7, n, payload, status=-1)
        c.call(7)  # valid probe must also leave the populated KV intact
        assert c.draft(12) == expected
        results.append({'case': 'five_invalid_requests_and_probe_preserve_kv', 'bitwise_equal': True})
        print(json.dumps({'passed': True, 'cases': results}, indent=2))
    finally:
        c.call(6, 0)
        c.call(2)
        c.sock.close()


if __name__ == '__main__':
    main()
