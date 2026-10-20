#!/usr/bin/env bash
set -euo pipefail

HPBEATS="hjotha@192.168.1.193"
DXC_EXE='C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe'
REMOTE_DIR='C:\Users\hjotha\temp_dxc'

args=()
src=""
out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -Fo)
            out="$2"
            shift 2
            ;;
        *.hlsl)
            src="$1"
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$src" || -z "$out" ]]; then
    echo "Usage: remote-dxc.sh [flags] <source.hlsl> -Fo <output.dxil>" >&2
    exit 1
fi

src_name="$(basename "$src")"
out_name="out_${RANDOM}_${src_name}.dxil"
remote_src="$REMOTE_DIR\\$src_name"
remote_out="$REMOTE_DIR\\$out_name"

scp -q "$src" "$HPBEATS:C:/Users/hjotha/temp_dxc/$src_name"
ssh "$HPBEATS" "\"$DXC_EXE\" ${args[*]} -I \"$REMOTE_DIR\" \"$remote_src\" -Fo \"$remote_out\""
scp -q "$HPBEATS:C:/Users/hjotha/temp_dxc/$out_name" "$out"
ssh "$HPBEATS" "cmd /c del \"$remote_out\" 2>nul"
