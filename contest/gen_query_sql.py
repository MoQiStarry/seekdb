#!/usr/bin/env python3
"""Emit N single-row approximate kNN SQL statements for QPS measurement.

Usage: gen_query_sql.py <query.f32> <dim> <nq> <topk> <out.sql>
"""
import struct
import sys

qf, dim, nq, topk, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
raw = open(qf, "rb").read()
avail = len(raw) // (dim * 4)
nq = min(nq, avail)
with open(out, "w") as f:
    f.write("use benchdb;\n")
    for i in range(nq):
        v = struct.unpack_from("<%df" % dim, raw, i * dim * 4)
        lit = "[" + ",".join("%.7f" % x for x in v) + "]"
        f.write("select id from items order by l2_distance(v,%s) approximate limit %d;\n" % (lit, topk))
print("wrote %d statements to %s" % (nq, out))
