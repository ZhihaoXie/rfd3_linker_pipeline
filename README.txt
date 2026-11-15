rfd3_linker_pipeline.py - 多链结构 -> RFdiffusion3 contig 设计 linker -> ProteinMPNN 重设计 linker

用法
  python rfd3_linker_pipeline.py -pdb <结构.pdb|.cif> -o <输出目录> [-l 10|5-15]
         [--segments A1-544,B31-730,C139-490,...]   # 省略则每链全段
         [--non-loopy true|false]                   # 默认 false（is_non_loopy=false，柔性 linker）；true 为刚性 linker
         [--gpu 0] [--num-seq 10] [--mpnn-temp 0.1] [--seed 42]
         [--n-batches N] [--batch-size M]
         [--reuse-rfd3 <已有 rfd3 输出目录>] [--reuse-mpnn] [--skip-mpnn]

输出
  <tag>_seq.fasta        序列文件
  <tag>_seq_info.csv     序列名称 / 序列总长度 / linker长度 / linkerN position / linkerN / 氨基酸序列
                         （列数随 segment 数动态：N 段 -> N-1 个 linker）
  <tag>_manifest.json    contig、segment 清单、每个设计的 linker 位置与骨架一致性
  rfd3_out/              RFdiffusion3 设计结构（cif.gz + json）
  mpnn/<设计名>/seqs/    ProteinMPNN 输出

环境
  RFdiffusion3 需 LD_LIBRARY_PATH=<envs/rfd3>/lib 且 HOME 指向含 .foundry/checkpoints 的账号目录
  （脚本用 RFD3_ENV / RFD3_HOME 环境变量处理，默认 /data/xiezhihao）。
