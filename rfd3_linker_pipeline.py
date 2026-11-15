#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rfd3_linker_pipeline.py - multi-chain structure -> RFdiffusion3 contig linker design
-> ProteinMPNN linker redesign.

Pipeline
  1) Read one PDB (or mmCIF) with multiple chains; cut the framework segments to keep.
  2) Build the RFdiffusion3 config json (contig = seg1,<linker>,seg2,<linker>,...,segN)
     and run `rfd3 design` to fuse the segments into one chain with designed linkers.
  3) On every design, locate the framework segments and derive each linker's
     position / length / sequence.
  4) Write the ProteinMPNN fixed_positions jsonl (framework fixed, linkers free)
     and run protein_mpnn_run.py to redesign the linker sequences.
  5) Output: <tag>_seq.fasta, <tag>_seq_info.csv (dynamic linker columns), <tag>_manifest.json.

Examples
  python rfd3_linker_pipeline.py -pdb design.pdb -l 10 --non-loopy true -o out/
  python rfd3_linker_pipeline.py -pdb design.pdb -l 5-15 --non-loopy false -o out/
  python rfd3_linker_pipeline.py -pdb design.pdb --segments A1-544,B31-730,C139-490 \\
         -l 10 -o out/ --reuse-rfd3 out/rfd3_out       # reuse finished rfd3 output
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time

# ---- environment (override with env vars) ----
RFD3_BIN = os.environ.get('RFD3_BIN', '/data/xiezhihao/software/miniconda3/envs/rfd3/bin/rfd3')
RFD3_ENV = os.environ.get('RFD3_ENV', '/data/xiezhihao/software/miniconda3/envs/rfd3')
RFD3_HOME = os.environ.get('RFD3_HOME', '/data/xiezhihao')  # weights live in $HOME/.foundry/checkpoints
MPNN_RUN = os.environ.get('MPNN_RUN', '/data/xiezhihao/software/ProteinMPNN/protein_mpnn_run.py')
MPNN_PY = os.environ.get('MPNN_PY', '/data/xiezhihao/software/miniconda3/envs/p312/bin/python')

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}
AA3.update({'MSE': 'M', 'SEC': 'U', 'PYL': 'O'})
AA1_3 = dict((v, k) for k, v in AA3.items())


def log(msg):
    sys.stdout.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def run(cmd, cwd=None, env=None):
    log('RUN: %s' % ' '.join(cmd))
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stdout or '')[-2000:] + '\n' + (r.stderr or '')[-2000:] + '\n')
        raise SystemExit('Command failed (rc=%d): %s' % (r.returncode, ' '.join(cmd)))
    return r.stdout


# ---------------- structure parsing ----------------
def read_text(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as fh:
            return fh.read()
    return open(path).read()


def pdb_chain_seqs(path):
    """{chain: [(resi, icode, aa1), ...]} from a PDB file, N->C order."""
    chains = {}
    for line in read_text(path).splitlines():
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        if line[12:16].strip() != 'CA':
            continue
        aa = AA3.get(line[17:20].strip())
        if aa is None:
            continue
        chains.setdefault(line[21], {})[(int(line[22:26]), line[26])] = aa
    return dict((ch, [(k[0], k[1], d[k]) for k in sorted(d)]) for ch, d in chains.items())


def cif_atom_records(path):
    """Parse the _atom_site loop of an mmCIF file into a list of dicts."""
    text = read_text(path)
    lines = text.splitlines()
    cols, recs, in_loop = [], [], False
    for i, line in enumerate(lines):
        s = line.strip()
        if s == 'loop_':
            in_loop = True
            cols = []
            continue
        if in_loop and s.startswith('_atom_site.'):
            cols.append(s[len('_atom_site.'):])
            continue
        if in_loop and cols:
            if s.startswith('#') or s.startswith('loop_') or s.startswith('_'):
                break
            if not s:
                continue
            parts = s.split()
            if len(parts) < len(cols):
                continue
            recs.append(dict(zip(cols, parts)))
    return cols, recs


def cif_chain_seqs(path):
    """{chain: [(resi, aa1), ...]} plus full atom rows, from mmCIF."""
    cols, recs = cif_atom_records(path)
    chains = {}
    for r in recs:
        if r.get('group_PDB') != 'ATOM':
            continue
        if r.get('label_atom_id') != 'CA':
            continue
        comp = r.get('auth_comp_id') or r.get('label_comp_id')
        aa = AA3.get(comp)
        if aa is None:
            continue
        ch = r.get('auth_asym_id') or r.get('label_asym_id')
        seqid = r.get('auth_seq_id') or r.get('label_seq_id')
        try:
            resi = int(float(seqid))
        except (TypeError, ValueError):
            continue
        chains.setdefault(ch, {}).setdefault(resi, aa)
    return dict((ch, [(k, ' ', d[k]) for k in sorted(d)]) for ch, d in chains.items())


def structure_chains(path):
    if path.endswith('.cif') or path.endswith('.cif.gz'):
        return cif_chain_seqs(path)
    return pdb_chain_seqs(path)


def cif_to_pdb(cif_path, pdb_path):
    """Convert an mmCIF (rfd3 output) to a minimal PDB for ProteinMPNN."""
    cols, recs = cif_atom_records(cif_path)
    order = {'N': 1, 'CA': 2, 'C': 3, 'O': 4, 'CB': 5}
    out = []
    for r in recs:
        if r.get('group_PDB') != 'ATOM':
            continue
        atom = r.get('label_atom_id') or r.get('auth_atom_id')
        comp = r.get('auth_comp_id') or r.get('label_comp_id')
        ch = r.get('auth_asym_id') or r.get('label_asym_id')
        resi = r.get('auth_seq_id') or r.get('label_seq_id')
        try:
            x = float(r['Cartn_x'])
            y = float(r['Cartn_y'])
            z = float(r['Cartn_z'])
            ri = int(float(resi))
        except (KeyError, TypeError, ValueError):
            continue
        name = atom if len(atom) == 4 else ' %-3s' % atom
        b = r.get('B_iso_or_equiv', '0')
        try:
            bf = float(b)
        except ValueError:
            bf = 0.0
        out.append((ch, ri, order.get(atom, 99), atom,
                    'ATOM  %5d %s %-3s %s%4d    %8.3f%8.3f%8.3f  1.00%8.2f          %2s'
                    % (0, name, comp, ch, ri, x, y, z, bf, 'C')))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    with open(pdb_path, 'w') as fh:
        for i, t in enumerate(out, 1):
            fh.write(t[4].replace('ATOM      0', 'ATOM  %5d' % i, 1)[:80].rstrip() + '\n')
        fh.write('END\n')
    return len(out)


def read_fasta(path):
    out, n, buf = [], None, []
    for line in open(path):
        if line.startswith('>'):
            if n is not None:
                out.append((n, ''.join(buf)))
            n = line[1:].strip()
            buf = []
        else:
            buf.append(line.strip().replace('*', ''))
    if n is not None:
        out.append((n, ''.join(buf)))
    return out


# ---------------- segments ----------------
def parse_segments(spec, chains):
    """spec 'A1-544,B31-730' or None (= every chain whole). -> [(label, chain, start, end, seq)]"""
    segs = []
    if spec:
        for part in spec.split(','):
            m = re.fullmatch(r'([A-Za-z])(\d+)-(\d+)', part.strip())
            if not m:
                raise SystemExit('segment format error: %s (expected e.g. A1-544)' % part)
            ch, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            if ch not in chains:
                raise SystemExit('chain %s not found in structure' % ch)
            sub = [x for x in chains[ch] if a <= x[0] <= b]
            if not sub:
                raise SystemExit('chain %s has no residues in %d-%d' % (ch, a, b))
            segs.append(('%s%d-%d' % (ch, a, b), ch, a, b, ''.join(x[2] for x in sub)))
    else:
        for ch in sorted(chains):
            r = chains[ch]
            a, b = r[0][0], r[-1][0]
            segs.append(('%s%d-%d' % (ch, a, b), ch, a, b, ''.join(x[2] for x in r)))
        if len(segs) < 2:
            raise SystemExit('at least 2 segments (2 chains) are required to design a linker')
    for lbl, ch, a, b, s in segs:
        log('segment %-14s %d aa' % (lbl, len(s)))
    return segs


# ---------------- RFdiffusion3 ----------------
def write_rfd3_json(segs, linker_spec, non_loopy, struct_path, out_json):
    parts = []
    for i, (lbl, ch, a, b, s) in enumerate(segs):
        parts.append('%s%d-%d' % (ch, a, b))
        if i < len(segs) - 1:
            parts.append(linker_spec)
    contig = ','.join(parts)
    fixed = dict(('%s%d-%d' % (ch, a, b), 'BKBN') for lbl, ch, a, b, s in segs)
    cfg = {'linker_design': {'input': os.path.abspath(struct_path),
                             'contig': contig,
                             'is_non_loopy': bool(non_loopy),
                             'select_fixed_atoms': fixed}}
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(cfg, fh, indent=2)
    log('RFdiffusion3 contig: %s (is_non_loopy=%s, %d linkers)'
        % (contig, non_loopy, len(segs) - 1))
    return contig


def run_rfd3(json_path, out_dir, n_batches, batch_size, gpu, workdir):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = os.path.join(RFD3_ENV, 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['HOME'] = RFD3_HOME
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cmd = [RFD3_BIN, 'design', 'inputs=%s' % os.path.abspath(json_path),
           'out_dir=%s' % os.path.abspath(out_dir),
           'n_batches=%d' % n_batches,
           'diffusion_batch_size=%d' % batch_size,
           'low_memory_mode=True']
    run(cmd, cwd=workdir, env=env)


def design_sequences(out_dir):
    """Collect rfd3 designs as (stem, seq, structure_path); convert CIF to PDB on demand."""
    res = []
    seen = set()
    for fa in sorted(glob.glob(os.path.join(out_dir, '*.fasta'))):
        stem = os.path.splitext(os.path.basename(fa))[0]
        recs = read_fasta(fa)
        if not recs:
            continue
        pdb = os.path.join(out_dir, stem + '.pdb')
        if not os.path.exists(pdb):
            for ext in ('.cif.gz', '.cif'):
                if os.path.exists(os.path.join(out_dir, stem + ext)):
                    cif_to_pdb(os.path.join(out_dir, stem + ext), pdb)
                    break
        res.append((stem, recs[0][1], pdb if os.path.exists(pdb) else None))
        seen.add(stem)
    for path in sorted(glob.glob(os.path.join(out_dir, '*.pdb'))
                       + glob.glob(os.path.join(out_dir, '*.cif'))
                       + glob.glob(os.path.join(out_dir, '*.cif.gz'))):
        stem = re.sub(r'\.(pdb|cif|cif\.gz)$', '', os.path.basename(path))
        if stem in seen:
            continue
        ch = structure_chains(path)
        if not ch:
            continue
        seq = ''.join(x[2] for x in ch[sorted(ch)[0]])
        pdb = os.path.join(out_dir, stem + '.pdb')
        if path.endswith('.cif') or path.endswith('.cif.gz'):
            if not os.path.exists(pdb):
                cif_to_pdb(path, pdb)
            struct = pdb
        else:
            struct = path
        res.append((stem, seq, struct))
        seen.add(stem)
    return res


# ---------------- linker location ----------------
def best_window(seq, seg, lo, hi):
    n = len(seg)
    best = (-1, -1)
    for i in range(lo, max(lo, min(hi, len(seq) - n) + 1)):
        idn = sum(1 for a, b in zip(seq[i:i + n], seg) if a == b)
        if idn > best[0]:
            best = (idn, i)
    return best[1], (best[0] / n if n else 0.0)


def locate(seq, segs, min_id=0.85):
    hits, prev = [], 0
    for k, (lbl, ch, a, b, s) in enumerate(segs):
        j = seq.find(s, prev)
        if j >= 0:
            hit, idn = j, 1.0
        else:
            hit, idn = best_window(seq, s, prev, len(seq) - len(s))
        if hit < 0 or idn < min_id:
            raise SystemExit('segment %s not found in design sequence (best identity %.2f)'
                             % (lbl, idn))
        hits.append([hit, len(s), idn])
        prev = hit + len(s)
    linkers = []
    for k in range(len(hits) - 1):
        st = hits[k][0] + hits[k][1]
        en = hits[k + 1][0]
        linkers.append((st, en, seq[st:en]))
    return hits, linkers


def linker_table(seq, segs, lmin, lmax):
    hits, linkers = locate(seq, segs)
    head = hits[0][0]
    tail = len(seq) - (hits[-1][0] + hits[-1][1])
    if head or tail:
        raise SystemExit('design sequence has %d/%d extra residues at the ends' % (head, tail))
    for k, (st, en, lk) in enumerate(linkers):
        if not (lmin <= len(lk) <= lmax):
            raise SystemExit('linker%d length %d is outside the requested range %d-%d'
                             % (k + 1, len(lk), lmin, lmax))
    pos = ['%d-%d' % (st + 1, en) for st, en, _ in linkers]
    return linkers, pos, hits


# ---------------- ProteinMPNN ----------------
def write_fixed_jsonl(seq_len, linkers, stem, out_jsonl):
    fixed, prev = [], 1
    for st, en, _ in linkers:
        fixed.extend(range(prev, st + 1))
        prev = en + 1
    fixed.extend(range(prev, seq_len + 1))
    with open(out_jsonl, 'w') as fh:
        fh.write(json.dumps({stem: {'A': fixed}}) + '\n')
    return len(fixed)


def run_mpnn(pdb_path, out_folder, fixed_jsonl, num_seq, temp, seed, batch_size):
    os.makedirs(out_folder, exist_ok=True)
    cmd = [MPNN_PY, MPNN_RUN,
           '--pdb_path', os.path.abspath(pdb_path),
           '--out_folder', os.path.abspath(out_folder),
           '--num_seq_per_target', str(num_seq),
           '--sampling_temp', str(temp),
           '--seed', str(seed),
           '--batch_size', str(batch_size),
           '--fixed_positions_jsonl', os.path.abspath(fixed_jsonl)]
    run(cmd)


def collect_mpnn(out_folder, stem):
    fa = os.path.join(out_folder, 'seqs', stem + '.fa')
    if not os.path.exists(fa):
        raise SystemExit('ProteinMPNN output missing: %s' % fa)
    recs = read_fasta(fa)
    designs = [(h, s) for h, s in recs if 'T=' in h or 'sample=' in h] or recs[1:]
    return [('%s_seq%d' % (stem, i + 1), s) for i, (h, s) in enumerate(designs)]


# ---------------- main ----------------
def parse_bool(value):
    """argparse type: true/false (also accepts yes/no, 1/0)."""
    v = str(value).strip().lower()
    if v in ('true', 't', 'yes', 'y', '1'):
        return True
    if v in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('expected true/false, got %r' % value)


def main():
    ap = argparse.ArgumentParser(
        description='Fuse multi-chain segments with RFdiffusion3 contig linkers, '
                    'then redesign the linkers with ProteinMPNN.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('-pdb', '--input', required=True,
                    help='input structure (PDB or mmCIF); each chain is a segment')
    ap.add_argument('-l', '--linker-length', default='10',
                    help='linker length: fixed value (5) or a range (5-15). Default 10')
    ap.add_argument('--non-loopy', dest='non_loopy', type=parse_bool, default=False,
                    metavar='true|false',
                    help='true = is_non_loopy=true, rigid/structured linker; '
                         'false = is_non_loopy=false, flexible linker (default)')
    ap.add_argument('--segments', default=None,
                    help='explicit framework segments in order, e.g. A1-544,B31-730,C139-490')
    ap.add_argument('-o', '--outdir', required=True, help='output directory')
    ap.add_argument('--tag', default=None, help='output file prefix (default: input basename)')
    ap.add_argument('--num-seq', type=int, default=10,
                    help='ProteinMPNN sequences per design (default 10)')
    ap.add_argument('--mpnn-temp', default='0.1', help='ProteinMPNN sampling_temp (default 0.1)')
    ap.add_argument('--seed', type=int, default=42, help='ProteinMPNN seed (default 42)')
    ap.add_argument('--n-batches', type=int, default=1, help='RFdiffusion3 n_batches (default 1)')
    ap.add_argument('--batch-size', type=int, default=1,
                    help='RFdiffusion3 diffusion_batch_size (default 1)')
    ap.add_argument('--mpnn-batch-size', type=int, default=1)
    ap.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES (default: unset)')
    ap.add_argument('--reuse-rfd3', default=None, metavar='DIR',
                    help='skip the RFdiffusion3 run and read designs from this existing out_dir')
    ap.add_argument('--reuse-mpnn', action='store_true',
                    help='reuse existing ProteinMPNN output under outdir/mpnn instead of rerunning it')
    ap.add_argument('--skip-mpnn', action='store_true',
                    help='run RFdiffusion3 only, skip ProteinMPNN')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit('input structure not found: %s' % args.input)
    m = re.fullmatch(r'(\d+)(?:-(\d+))?', str(args.linker_length).strip())
    if not m:
        raise SystemExit('-l format error: %s (expected e.g. 5 or 5-15)' % args.linker_length)
    lmin = int(m.group(1))
    lmax = int(m.group(2) or m.group(1))
    if lmax < lmin:
        raise SystemExit('-l lower bound is greater than the upper bound')
    linker_spec = str(args.linker_length).strip()

    os.makedirs(args.outdir, exist_ok=True)
    work = os.path.join(args.outdir, 'work')
    os.makedirs(work, exist_ok=True)
    tag = args.tag or re.sub(r'\.(pdb|cif|cif\.gz)$', '',
                             os.path.basename(args.input))

    chains = structure_chains(args.input)
    if not chains:
        raise SystemExit('no protein chains parsed from the input structure')
    log('input %s chains: %s' % (args.input,
                                 ', '.join('%s(%d)' % (c, len(v)) for c, v in sorted(chains.items())))
        )
    segs = parse_segments(args.segments, chains)
    n_linkers = len(segs) - 1

    # 1) RFdiffusion3
    rfd_json = os.path.join(work, 'rfd3_input.json')
    contig = write_rfd3_json(segs, linker_spec, args.non_loopy, args.input, rfd_json)
    rfd_out = os.path.join(args.outdir, 'rfd3_out')
    if args.reuse_rfd3:
        rfd_out = os.path.abspath(args.reuse_rfd3)
        if not os.path.isdir(rfd_out):
            raise SystemExit('--reuse-rfd3 directory not found: %s' % rfd_out)
        log('reusing existing RFdiffusion3 output: %s' % rfd_out)
    else:
        run_rfd3(rfd_json, rfd_out, args.n_batches, args.batch_size, args.gpu, work)
    designs = design_sequences(rfd_out)
    if not designs:
        raise SystemExit('RFdiffusion3 produced no design (no .pdb/.cif/.fasta in %s)' % rfd_out)
    log('collected %d design(s), %d linker(s) each' % (len(designs), n_linkers))

    # 2) linker location + ProteinMPNN
    fasta_recs, rows, manifest = [], [], []
    for stem, dseq, struct in designs:
        linkers, pos, hits = linker_table(dseq, segs, lmin, lmax)
        log('%s length=%d linkers: %s (segment identity %s)'
            % (stem, len(dseq),
               ' / '.join('%s(%d)' % (p, len(l)) for p, (_, _, l) in zip(pos, linkers)),
               ', '.join('%.3f' % h[2] for h in hits)))
        manifest.append({'design': stem, 'length': len(dseq),
                         'linkers': [{'position': p, 'length': len(l), 'sequence': l}
                                     for p, (_, _, l) in zip(pos, linkers)],
                         'segment_identity': [round(h[2], 3) for h in hits]})

        recs = [(stem, dseq)]
        if not args.skip_mpnn:
            if not struct or not os.path.exists(struct):
                raise SystemExit('design structure PDB missing (required by ProteinMPNN): %s' % struct)
            fj = os.path.join(work, stem + '.fixed.jsonl')
            nfix = write_fixed_jsonl(len(dseq), linkers, stem, fj)
            log('%s fixed %d positions, %d linker positions left for ProteinMPNN'
                % (stem, nfix, len(dseq) - nfix))
            mpnn_dir = os.path.join(args.outdir, 'mpnn', stem)
            done = os.path.exists(os.path.join(mpnn_dir, 'seqs', stem + '.fa'))
            if args.reuse_mpnn and done:
                log('%s reusing existing ProteinMPNN output' % stem)
            else:
                run_mpnn(struct, mpnn_dir, fj,
                         args.num_seq, args.mpnn_temp, args.seed, args.mpnn_batch_size)
            recs = collect_mpnn(mpnn_dir, stem)
        fasta_recs.extend(recs)

        for name, s in recs:
            if len(s) != len(dseq):
                raise SystemExit('%s length %d differs from design length %d'
                                 % (name, len(s), len(dseq)))
            linkers_s, pos2, _ = linker_table(s, segs, lmin, lmax)
            if pos2 != pos:
                raise SystemExit('%s linker positions shifted after redesign' % name)
            lks = [x[2] for x in linkers_s]
            row = {'序列名称': name, '序列总长度': len(s),
                   'linker长度': sum(len(x) for x in lks)}
            for i in range(n_linkers):
                row['linker%d position' % (i + 1)] = pos[i]
            for i in range(n_linkers):
                row['linker%d' % (i + 1)] = lks[i]
            row['氨基酸序列'] = s
            rows.append(row)

    fields = ['序列名称', '序列总长度', 'linker长度'] \
             + ['linker%d position' % (i + 1) for i in range(n_linkers)] \
             + ['linker%d' % (i + 1) for i in range(n_linkers)] \
             + ['氨基酸序列']

    fa_path = os.path.join(args.outdir, '%s_seq.fasta' % tag)
    with open(fa_path, 'w') as fh:
        for n, s in fasta_recs:
            fh.write('>%s\n%s\n' % (n, s))

    csv_path = os.path.join(args.outdir, '%s_seq_info.csv' % tag)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(args.outdir, '%s_manifest.json' % tag), 'w') as fh:
        json.dump({'input': os.path.abspath(args.input),
                   'segments': [l[0] for l in segs],
                   'n_linkers': n_linkers,
                   'linker_length': linker_spec,
                   'is_non_loopy': args.non_loopy,
                   'contig': contig,
                   'rfd3_out': os.path.abspath(rfd_out),
                   'mpnn': not args.skip_mpnn,
                   'designs': manifest,
                   'n_sequences': len(fasta_recs)}, fh, indent=2, ensure_ascii=False)

    log('done: %d sequence(s), %d linker(s) per sequence' % (len(fasta_recs), n_linkers))
    log('fasta: %s' % fa_path)
    log('csv  : %s' % csv_path)


if __name__ == '__main__':
    main()
