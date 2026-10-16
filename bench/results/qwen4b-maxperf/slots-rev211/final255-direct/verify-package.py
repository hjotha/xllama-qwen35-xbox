from pathlib import Path
import zipfile,hashlib,json,xml.etree.ElementTree as ET,re
b=Path(__file__).resolve().parent;p=b/'xllama_1.6.0.255_x64.msix'
h=lambda x:hashlib.sha256(x).hexdigest()
log=(b/'build-local.log').read_text(encoding='utf-8-sig')
expected=re.findall(r'sha256-fixed=([0-9A-Fa-f]{64})',log);assert len(expected)==1 and h(p.read_bytes())==expected[0].lower()
assert 'mode=final' in log and 'signer=21E98358E40980181C8795919FE3C7D455E2B7FC' in log
with zipfile.ZipFile(p) as z:
 i=ET.fromstring(z.read('AppxManifest.xml')).find('{*}Identity').attrib
 assert i['Version']=='1.6.0.255' and i['Publisher']=='CN=xllama-dev' and i['Name']=='GianlucaMazza.xllama'
 crt={n:h(z.read(n)) for n in ('MSVCP140.dll','MSVCP140_1.dll','VCRUNTIME140.dll','VCRUNTIME140_1.dll')}
 assert crt==json.loads((b.parent/'final251-direct/package-verification.json').read_text())['crt']
 exe=z.read('xllama.exe')
 v={'sha256':h(p.read_bytes()),'bytes':p.stat().st_size,'identity':i,'signature_sha256':h(z.read('AppxSignature.p7x')),'crt':crt,'mode':'Final/LTCG','build_job':'BjO58pV7sZsmbdTGw5-MkBmkmI71K1fu','exe_sha256':h(exe),'exe_bytes':len(exe)}
raw=(b/'msbuild.out').read_bytes();text=raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8-sig')
cls=[x for x in text.splitlines() if re.search(r'CL\.exe\s+/c\s',x,re.I)]
links=[x for x in text.splitlines() if re.search(r'link\.exe\s',x,re.I)]
assert cls and links
assert all(re.search(r'\s/GL(?:\s|$)',x,re.I) for x in cls)
assert all(re.search(r'\s/LTCG(?:\s|$)',x,re.I) for x in links)
assert any('ops.cpp' in x for x in cls)
linked=json.loads((b/'linked-exe-info.json').read_text(encoding='utf-8-sig'))
assert linked['sha256'].lower()==h(exe) and linked['bytes']==len(exe)
proof={'mode':'Final/LTCG','msbuild_log_sha256':h(raw),'current_CL_invocations':len(cls),'GL_enabled':len(cls),'current_link_invocations':len(links),'LTCG_enabled':len(links),'exe_sha256':h(exe),'exe_bytes':len(exe),'commands':cls+links,'cache':'Release, separated from Release-iteration','linked_exe':linked}
(b/'final-build-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
(b/'package-verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2));print('Current compile/link proof',len(cls),'GL,',len(links),'LTCG')
for parent in [b,b.parent,b.parent.parent]:assert not (parent/'xllama-test.cer').exists()
