import csv, glob, hashlib, os, sys, statistics
D=sys.argv[1]
DIG={"chat64":"84e081965076a0c947bb9d45953ec5b8768c25f79db37075a933553c000129f2",
"code64":"ff42c5beec71ff491b458d0e48cbac4d98f8cdcf8153c78868c64aee687a4010",
"chat256":"f78ce8372ccc7281aaad5e726bf7c88a0317212e3b2ed397fb0fd83ffd53066c",
"code256":"c7d84255e08e20776c1ec6743d23161d32d08dcba642036b305250f10841ec0c",
"std512":"44546453971b25da9afb886c6dd0c246fb49fff5069aaa444b099108bd0e8732"}
NEW={"std64-mtp","std256","std256-mtp"}  # no historical baseline: arm-parity gate only
files=sorted(glob.glob(D+"/csv/*.csv"))
fails=[]; t0={}; t1={}; dig={}
for f in files:
    name=os.path.basename(f)[:-4]
    cell,sws=name.rsplit("-sw",1); sw=int(sws[0])
    rows=list(csv.DictReader(open(f)))
    idx=[r["run_index"] for r in rows]
    if not(len(rows)==2 and set(idx)=={"2","3"}):
        fails.append(f"ROW {name} rows={len(rows)} idx={idx}")
    basekey = "std512" if cell=="std64" else cell
    for r in (2,3):
        p=f.replace(".csv",f".run{r}.tokens")
        if not os.path.exists(p): fails.append(f"TOK_MISSING {name} run{r}"); continue
        h=hashlib.sha256(open(p,'rb').read()).hexdigest()
        dig[(cell,sw,r)]=h
        if cell not in NEW and basekey in DIG and h!=DIG[basekey]:
            fails.append(f"BASE_DIGEST_FAIL {name} run{r} got={h[:16]}")
    dec=[float(r["decode_tok_s"]) for r in rows]
    (t0 if sw==0 else t1).setdefault(cell,[]).extend(dec)
for cell in sorted({c for (c,_,_) in dig}):
    for r in (2,3):
        a=dig.get((cell,0,r)); b=dig.get((cell,1,r))
        if a and b and a!=b: fails.append(f"ARM_PARITY_FAIL {cell} run{r} {a[:16]} vs {b[:16]}")
print(f"== {D}: failures={len(fails)}")
for x in fails: print("  ",x)
print(f"{'cell':<12} {'sw0_mean':>8} {'sw1_mean':>8} {'delta%':>7}")
for k in sorted(set(t0)|set(t1)):
    a=t0.get(k,[]); b=t1.get(k,[])
    if a and b:
        print(f"{k:<12} {statistics.mean(a):8.2f} {statistics.mean(b):8.2f} {100*(statistics.mean(b)/statistics.mean(a)-1):+6.2f}%")
