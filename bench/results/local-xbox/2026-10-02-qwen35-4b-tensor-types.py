import struct,collections
f=open('Qwen3.5-4B-Q4_K_M.gguf','rb'); rd=f.read
def u64(): return struct.unpack('<Q',rd(8))[0]
def u32(): return struct.unpack('<I',rd(4))[0]
def i64(): return struct.unpack('<q',rd(8))[0]
def s():
    n=u64(); return rd(n).decode('utf-8',errors='replace')
def skip(t):
    if t==0: rd(1)
    elif t==1: rd(1)
    elif t==2: rd(2)
    elif t==3: rd(2)
    elif t in (4,5,6): rd(4)
    elif t==7: rd(1)
    elif t==8: n=u64(); rd(n)
    elif t==10: rd(8)
    elif t==11: rd(8)
    elif t==12: rd(8)
    elif t==9:
        et=u32(); n=u64()
        for _ in range(n): skip(et)
    else: raise ValueError('kvtype %d'%t)
magic=rd(4); ver=u32(); n_t=u64(); n_kv=u64()
print('magic',magic,'ver',ver,'tensors',n_t,'kv',n_kv)
kv={}
for _ in range(n_kv):
    k=s(); t=u32()
    if t==8: kv[k]=s()
    elif t==0: kv[k]=struct.unpack('<B',rd(1))[0]
    elif t==1: kv[k]=struct.unpack('<b',rd(1))[0]
    elif t==4: kv[k]=struct.unpack('<I',rd(4))[0]
    elif t==5: kv[k]=struct.unpack('<i',rd(4))[0]
    elif t==6: kv[k]=struct.unpack('<f',rd(4))[0]
    elif t==7: kv[k]=bool(rd(1)[0])
    elif t==10: kv[k]=u64()
    elif t==11: kv[k]=i64()
    elif t==12: kv[k]=struct.unpack('<d',rd(8))[0]
    elif t==9:
        et=u32(); n=u64(); v=[]
        for i in range(n):
            if et==8: v.append(s())
            elif et==4: v.append(struct.unpack('<I',rd(4))[0])
            elif et==0: v.append(struct.unpack('<B',rd(1))[0])
            else: skip(et); v.append(None)
        kv[k]=v
    else: raise ValueError('kvtype %d'%t)
for k in sorted(kv):
    if k.startswith('qwen35') or k.startswith('general'):
        print('  %-44s = %s' % (k,kv[k]))
GT={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K'}
tn=collections.Counter(); names={}
for _ in range(n_t):
    nm=s(); nd=u32(); dims=[u64() for _ in range(nd)]; ty=u32(); u64()
    t=GT.get(ty,'t%d'%ty); tn[t]+=1; names.setdefault(t,[]).append((nm,dims))
print()
print('=== tipos de tensor ===')
for t,c in tn.most_common():
    ex=[n for n,_ in names[t][:3]]
    print('  %-6s x%-4d ex: %s' % (t,c,ex))
print()
print('=== K (ne0) distintos por tipo ===')
for t in sorted(names):
    ks=sorted({d[0] for _,d in names[t]})
    print('  %-6s K=%s' % (t,ks))

print()
BLK={'F32':1,'F16':2,'Q4_0':32,'Q8_0':32,'Q4_K':256,'Q5_K':256,'Q6_K':256,'Q2_K':256,'Q3_K':256,'Q8_K':256}
tot={}; cov=0; nocov=0; nocov_names={}
allt=[]
for t,lst in names.items():
    for nm,dims in lst:
        n=1
        for d in dims: n*=d
        b=(n//BLK.get(t,1))*(144 if t=='Q6_K' else (80 if t in ('Q4_K','Q5_K') else (34 if t=='Q8_0' else (18 if t=='Q4_0' else 4))))
        if t=='F32': b=n*4
        if t=='Q8_0': b=(n//32)*34
        if t in ('Q4_K','Q5_K'): b=(n//256)*84
        if t=='Q6_K': b=(n//256)*210
        tot[t]=tot.get(t,0)+b
        allt.append((nm,t,b,dims))
T=sum(tot.values())
print('=== bytes por tipo (total %.3f GB) ===' % (T/1e9))
for t,c in sorted(tot.items(), key=lambda x:-x[1]):
    tag='COBERTO' if t in ('Q4_K','Q6_K','Q4_0') else 'sem kernel'
    print('  %-6s %8.3f GB  %5.1f%%  %s' % (t,c/1e9,100*c/T,tag))
print()
print('=== pesos de matmul por tipo ===')
mm={}
for nm,t,b,dims in allt:
    if len(dims)==2 and t!='F32':
        mm[t]=mm.get(t,0)+b
M=sum(mm.values())
for t,c in sorted(mm.items(), key=lambda x:-x[1]):
    tag='COBERTO' if t in ('Q4_K','Q6_K','Q4_0') else '*** SEM KERNEL ***'
    print('  %-6s %8.3f GB  %5.1f%%  %s' % (t,c/1e9,100*c/M,tag))
print()
print('=== matmuls Q5_K/Q8_0 (os que ficariam na CPU) ===')
seen=set()
for nm,t,b,dims in sorted(allt, key=lambda x:-x[2]):
    if len(dims)==2 and t in ('Q5_K','Q8_0'):
        k=nm.split('.')[-1]
        if k in seen: continue
        seen.add(k)
        print('  %-28s %-5s %sx%s  %.1f MB' % (k,t,dims[0],dims[1],b/1e6))
