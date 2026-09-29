"""Small Device Portal client for this Xbox probe; no app-type changes."""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import ssl
import sys
import urllib.error
import urllib.request
import uuid


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("route")
    parser.add_argument("--method", default="GET", choices=["GET", "POST", "PUT", "DELETE"])
    parser.add_argument("--host", default="192.168.1.26")
    parser.add_argument("--upload", nargs="+")
    payload = parser.add_mutually_exclusive_group()
    payload.add_argument("--json")
    payload.add_argument("--json-file", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.route.startswith("/") or args.route.startswith("//"):
        parser.error("route must be an absolute Device Portal path")
    base = "https://" + args.host + ":11443"
    cookies = http.cookiejar.CookieJar()
    # Xbox Dev Mode uses a self-signed certificate on the user's LAN.
    client = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookies),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
    )
    headers = {}
    data = None
    if args.method != "GET":
        with client.open(base + "/", timeout=15) as response:
            response.read()
        for cookie in cookies:
            if cookie.name == "CSRF-Token":
                headers["X-CSRF-Token"] = cookie.value
    if args.upload:
        boundary = "XboxProbe" + uuid.uuid4().hex
        parts = []
        for file in map(Path, args.upload):
            parts.append(("--" + boundary + '\r\nContent-Disposition: form-data; name="' + file.name + '"; filename="' + file.name + '"\r\nContent-Type: application/octet-stream\r\n\r\n').encode())
            parts.extend([file.read_bytes(), b"\r\n"])
        parts.append(("--" + boundary + "--\r\n").encode())
        data = b"".join(parts)
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    elif args.json or args.json_file:
        raw_json = args.json_file.read_text(encoding="utf-8") if args.json_file else args.json
        data = json.dumps(json.loads(raw_json)).encode()
        headers["Content-Type"] = "application/json"
    elif args.method in ("POST", "PUT"):
        data = b""
    request = urllib.request.Request(base + args.route, data=data, headers=headers, method=args.method)
    try:
        with client.open(request, timeout=args.timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        print(json.dumps({"http_status": error.code, "error": error.read().decode(errors="replace")}))
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(body)
        print(json.dumps({"http_status": status, "path": str(args.out.resolve()), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}))
    else:
        try:
            value = json.loads(body)
        except ValueError:
            value = body.decode(errors="replace")
        print(json.dumps({"http_status": status, "data": value}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
