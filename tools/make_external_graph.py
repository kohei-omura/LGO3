#!/usr/bin/env python3
"""
ONNX モデルから「重みを含まない設計図(graph)」を作る。

重み(initializer の raw_data)は元の model.onnx の中にそのまま並んでいるので、
設計図の各重みを external_data として「元ファイル内のバイト位置」に向ける。
これで、配布元(Hugging Face)からダウンロードした未改変のモデルファイルを
そのまま重みの置き場として使える（再配布・再アップロード不要）。

onnxruntime-web は external data を JS 側に保持したまま GPU へ直接転送するため、
モデル全体を wasm メモリへ複製しない → iPhone のメモリ上限(約1.9GB)を超えにくくなる。

使い方: make_external_graph.py <元のmodel.onnx> <出力graph.onnx> <location名>
"""
import sys, onnx
from onnx import TensorProto

def varint(buf, i):
    r = s = 0
    while True:
        b = buf[i]; i += 1
        r |= (b & 0x7f) << s; s += 7
        if b < 0x80: return r, i

def fields(buf, start, end):
    """protobuf メッセージ [start,end) のフィールドを (番号, 型, 値の開始, 値の終了) で列挙"""
    i = start
    while i < end:
        key, i = varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0: _, j = varint(buf, i); yield num, wt, i, j; i = j
        elif wt == 1: yield num, wt, i, i + 8; i += 8
        elif wt == 5: yield num, wt, i, i + 4; i += 4
        elif wt == 2:
            ln, i = varint(buf, i); yield num, wt, i, i + ln; i += ln
        else: raise ValueError(f'unsupported wire type {wt}')

def initializer_offsets(buf):
    """ModelProto.graph(7).initializer(5) の各 TensorProto について name(8) と raw_data(9) の位置"""
    out = {}
    for num, wt, s, e in fields(buf, 0, len(buf)):
        if num != 7 or wt != 2: continue
        for gnum, gwt, gs, ge in fields(buf, s, e):
            if gnum != 5 or gwt != 2: continue
            name = raw = None
            for tnum, twt, ts, te in fields(buf, gs, ge):
                if tnum == 8 and twt == 2: name = bytes(buf[ts:te]).decode()
                elif tnum == 9 and twt == 2: raw = (ts, te - ts)
            if name is not None and raw is not None: out[name] = raw
    return out

def main(src, dst, location, min_bytes=1024):
    buf = memoryview(open(src, 'rb').read())
    offs = initializer_offsets(buf)
    m = onnx.load(src, load_external_data=False)
    moved = kept = moved_bytes = 0
    for t in m.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL or not t.HasField('raw_data') or len(t.raw_data) < min_bytes:
            kept += 1; continue
        off, ln = offs[t.name]
        assert ln == len(t.raw_data) and bytes(buf[off:off + 16]) == t.raw_data[:16], t.name
        t.ClearField('raw_data')
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (('location', location), ('offset', str(off)), ('length', str(ln))):
            e = t.external_data.add(); e.key = k; e.value = v
        moved += 1; moved_bytes += ln
    onnx.save(m, dst)
    print(f'{dst}: 外部化 {moved}個 ({moved_bytes/1048576:.1f}MB) / 内蔵 {kept}個 / 設計図 {len(open(dst,"rb").read())/1048576:.2f}MB')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
