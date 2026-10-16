import struct, pathlib, hashlib, json, datetime
root=pathlib.Path(__file__).resolve().parent
d=root/'cold-prefill'
arms={k:(d/k/'chat-dump.kv').read_bytes() for k in ('C','R','F')}
a,b=arms['C'],arms['F']; assert a==arms['R']
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in arms.items()},'C_R_equal':True,'attention':[],'recurrent':[]}
o=0
def take(fmt):
 global o
 v=struct.unpack_from('<'+fmt,a,o);assert a[o:o+struct.calcsize('<'+fmt)]==b[o:o+struct.calcsize('<'+fmt)]
 o+=struct.calcsize('<'+fmt);return v
magic,seq,streams,cells=take('4I');assert streams==2 and cells==229
for i in range(cells):
 pos,nseq,x,y,tok,sid=take('6i');assert pos==i and nseq==1 and sid==0
trans,layers=take('2I');assert trans==0 and layers==8
for kind in ('K','V'):
 for layer in range(layers):
  typ,row=take('iQ');assert typ==1 and row==2048
  diffs=[i for i in range(cells) if a[o+i*row:o+(i+1)*row]!=b[o+i*row:o+(i+1)*row]]
  out['attention'].append(dict(kind=kind,cache_layer=layer,offset=o,row_bytes=row,differing_positions=diffs))
  o+=cells*row
empty,ncell,pos,nseq,trans,nlayers=take('6I');assert (empty,ncell,pos,nseq,trans,nlayers)==(0,1,228,0,0,32)
# Qwen35 default full-attention interval 4. Verify mapping separately from model metadata.
for kind in ('R','S'):
 for il in range(nlayers):
  if il%4==3:continue
  typ,row=take('iQ');assert typ==0 and row in (98304,2097152)
  x,y=a[o:o+row],b[o:o+row]
  diffs=sum(x[j:j+4]!=y[j:j+4] for j in range(0,row,4))
  out['recurrent'].append(dict(kind=kind,model_layer_assuming_interval4=il,offset=o,row_bytes=row,differing_float32=diffs,total_float32=row//4))
  o+=row
assert o==len(a)==len(b)
out['parsed_to_eof']=o
out['interpretation']='All 229 attention positions 0..228 (sequence0), eight attention K/V layers, and 24 recurrent R/S layer states match byte for byte across C/R/F. Parsed complete native state; no tolerance.'
assert all(not x['differing_positions'] for x in out['attention'])
assert all(x['differing_float32']==0 for x in out['recurrent'])
out['attention_positions']=list(range(cells));out['recurrent_position']=pos
(root/'native-state-final255.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'parsed_bytes':o,'attention_positions':cells,'attention_KV_records':len(out['attention']),'recurrent_records':len(out['recurrent']),'recurrent_position':pos,'all_exact':True}))
