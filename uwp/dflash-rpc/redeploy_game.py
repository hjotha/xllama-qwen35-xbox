"""Temporarily set the Xbox UWP install default to Game for this test package."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SETTING = "/ext/settings/DefaultUWPContentTypeToGame"


def call(portal, route, *args):
    result = subprocess.run(
        [sys.executable, str(portal), route, *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"{route}: {result.stdout.strip()} {result.stderr.strip()}")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not args.portal.is_file() or not args.package.is_file():
        parser.error("portal script or signed package missing")

    events = []
    current = call(args.portal, SETTING)["data"]
    original = current["Value"]
    if original.lower() not in ("true", "false"):
        raise ValueError(f"Unexpected setting value: {original!r}")
    events.append({"step": "before", "setting": current})
    changed = False
    try:
        if original.lower() != "true":
            events.append({"step": "set_game", "reply": call(
                args.portal, SETTING, "--method", "PUT", "--json", json.dumps({"Value": "True"})
            )})
            changed = True
        events.append({"step": "during", "setting": call(args.portal, SETTING)["data"]})
        events.append({"step": "install", "reply": call(
            args.portal, f"/api/app/packagemanager/package?package={args.package.name}", "--method", "POST",
            "--upload", str(args.package)
        )})
        events.append({"step": "install_state", "reply": call(
            args.portal, "/api/app/packagemanager/state"
        )})
    finally:
        if changed:
            events.append({"step": "restore", "reply": call(
                args.portal, SETTING, "--method", "PUT", "--json", json.dumps({"Value": original.title()})
            )})
        events.append({"step": "after", "setting": call(args.portal, SETTING)["data"]})
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    if events[-1]["setting"]["Value"].lower() != original.lower():
        raise RuntimeError("Xbox default UWP type was not restored")
    install_state = next(event["reply"]["data"] for event in events if event["step"] == "install_state")
    if not install_state["Success"]:
        raise RuntimeError(f"Xbox deployment failed: {install_state}")
    print(json.dumps({"setting_before": original, "setting_after": events[-1]["setting"]["Value"],
                      "install_state": install_state, "log": str(args.log.resolve())}))


if __name__ == "__main__":
    main()
