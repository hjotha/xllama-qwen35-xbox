"""Rebuild the saved research baseline, then restore and rebuild optimized sources."""
from pathlib import Path
import shutil
import subprocess

root = Path('/home/hjotha/beellama.cpp')
work = Path('/tmp/dflash-trim-opt')
names = ('common/dflash-remote.h', 'common/speculative.cpp')
saved = {name: (root / name).read_bytes() for name in names}
build = ['cmake', '--build', str(root / 'build-dflash-xbox-dl'), '--target', 'llama-server', '-j', '4']


def compile_to(label):
    with (work / f'build-{label}.log').open('w') as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    destination = work / f'bin-{label}'
    destination.mkdir(exist_ok=True)
    source = root / 'build-dflash-xbox-dl/bin'
    for item in source.iterdir():
        if item.name == 'llama-server' or item.name.startswith(('libllama-common.so', 'libllama-server-impl.so')):
            shutil.copy2(item, destination / item.name)
        elif item.name.startswith('libggml-') and '.so' in item.name:
            link = destination / item.name
            if not link.exists():
                link.symlink_to(item)
    print(f'BUILD_AND_SNAPSHOT_OK={label}', flush=True)


try:
    for name in names:
        (root / name).write_bytes((work / 'before' / Path(name).name).read_bytes())
    compile_to('before')
finally:
    for name, data in saved.items():
        (root / name).write_bytes(data)
    compile_to('after')
