#!/usr/bin/env python3
"""
描画部品(unet)の設計図(graph)で、大きな中間結果を作る計算を「縦方向（画素の並び）に小分け」にする。
iPhone の Safari では 256×384 以上で強制終了が起きるため、同時に必要な GPU メモリを減らす。

1) 自己注意(attn1): 画素どうしの関係を表す表(scores)が 画素数×画素数 になり、448×640 では 1個 306MB。
   問い合わせ側(Q)を K 個に分け、Q_i → MatMul(Q_i, K) → Softmax → MatMul(・, V) を順に計算して連結する。
   Softmax は横方向（キー側）だけなので、分けても値は同じ。
2) 全結合(ff): 1画素ずつ独立した計算（拡大→GEGLU→縮小）なので、入力を画素の並びで K 個に分け、
   それぞれを同じ重みで計算して連結する。拡大後の中間（448×640 で 1個 22MB）が 1/K になる。

どちらも計算内容は元と同じ（WebGPU で出力の完全一致を確認）。
重み(initializer)は外部データ（元ファイル内のバイト位置）を指したまま。追加するのは
分割位置などの小さな定数だけで、元のモデルファイルはそのまま使える。

使い方: slice_unet.py <入力graph.onnx> <出力graph.onnx>
"""
import sys
import onnx
from onnx import helper, numpy_helper
import numpy as np

# tiny-sd: down_blocks.0 / up_blocks.2 が最大解像度、down_blocks.1 / up_blocks.1 が 1/2、残りは 1/4
def level_of(prefix):
    if prefix.startswith('/down_blocks.0/') or prefix.startswith('/up_blocks.2/'):
        return 0
    if prefix.startswith('/down_blocks.1/') or prefix.startswith('/up_blocks.1/'):
        return 1
    return 2


# 分割数（2の累乗）。448×640 で 自己注意の表 1個 ≤ 10MB、全結合の中間 1個 ≤ 6MB になるように選んだ
ATTN_CHUNKS = {0: 32, 1: 2, 2: 1}
FF_CHUNKS = {0: 4, 1: 2, 2: 1}


def const(g, name, vals):
    g.initializer.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), name))
    return name


def concat_tree(nodes, parts, out_name, base, axis, fan_in=4):
    """連結はバインド数の上限を避けるため 4 個ずつの木にする"""
    level = 0
    while len(parts) > fan_in:
        nxt = []
        for j in range(0, len(parts), fan_in):
            grp = parts[j:j + fan_in]
            if len(grp) == 1:
                nxt.append(grp[0])
                continue
            o = f'{base}/Concat_{level}_{j // fan_in}_output_0'
            nodes.append(helper.make_node('Concat', grp, [o], name=f'{base}/Concat_{level}_{j // fan_in}', axis=axis))
            nxt.append(o)
        parts, level = nxt, level + 1
    nodes.append(helper.make_node('Concat', parts, [out_name], name=f'{base}/Concat_final', axis=axis))


def chunk_bounds(g, nodes, x, axis, k, base):
    """x の axis 方向の長さ N を K 個に分ける位置（ceil(N/K) ずつ）。大きさが決まれば ORT が定数に畳み込む"""
    nodes.append(helper.make_node('Shape', [x], [base + '/N'], name=base + '/Shape', start=axis, end=axis + 1))
    nodes.append(helper.make_node('Add', [base + '/N', const(g, base + '/km1', [k - 1])], [base + '/Np'], name=base + '/Add'))
    nodes.append(helper.make_node('Div', [base + '/Np', const(g, base + '/k', [k])], [base + '/step'], name=base + '/Div'))
    axes = const(g, base + '/axes', [axis])
    out = []
    for i in range(k):
        c = f'{base}/c{i}'
        nodes.append(helper.make_node('Mul', [base + '/step', const(g, c + '/i0', [i])], [c + '/start'], name=c + '/Mul0'))
        nodes.append(helper.make_node('Mul', [base + '/step', const(g, c + '/i1', [i + 1])], [c + '/end'], name=c + '/Mul1'))
        nodes.append(helper.make_node('Slice', [x, c + '/start', c + '/end', axes], [c + '/x'], name=c + '/Slice'))
        out.append(c + '/x')
    return out


def slice_unet(model):
    g = model.graph
    byname = {n.name: n for n in g.node}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    outputs = {o.name for o in g.output}
    replace = {}      # id(置き換える位置のノード) → 追加ノードを作る関数
    drop = set()      # 取り除くノード
    gone = set()      # 消える中間結果の名前
    done = []

    # ── 1) 自己注意 ──
    for n in list(g.node):
        if n.op_type != 'Softmax' or '/attn1/' not in n.name:
            continue
        p = n.name.rsplit('/', 1)[0]
        k = ATTN_CHUNKS[level_of(p)]
        if k <= 1:
            continue
        mm, mm1 = byname.get(p + '/MatMul'), byname.get(p + '/MatMul_1')
        if mm is None or mm1 is None or n.input[0] != mm.output[0] or mm1.input[0] != n.output[0]:
            raise SystemExit(f'{p}: 想定と違う形です')
        for t in (mm.output[0], n.output[0]):
            if len(consumers.get(t, [])) != 1 or t in outputs:
                raise SystemExit(f'{p}: {t} が他でも使われています')

        def make(nodes, p=p, mm=mm, mm1=mm1, k=k):
            q, kt, v = mm.input[0], mm.input[1], mm1.input[1]
            b = p + '/slice'
            parts = []
            for i, qi in enumerate(chunk_bounds(g, nodes, q, 2, k, b)):
                c = f'{b}/c{i}'
                nodes.append(helper.make_node('MatMul', [qi, kt], [c + '/scores'], name=c + '/MatMul'))
                nodes.append(helper.make_node('Softmax', [c + '/scores'], [c + '/probs'], name=c + '/Softmax', axis=-1))
                nodes.append(helper.make_node('MatMul', [c + '/probs', v], [c + '/out'], name=c + '/MatMul_1'))
                parts.append(c + '/out')
            concat_tree(nodes, parts, mm1.output[0], b, 2)
        replace[id(mm1)] = make
        drop.update({id(mm), id(n)})
        gone.update({mm.output[0], n.output[0]})
        done.append(('自己注意', p, k))

    # ── 2) 全結合 ──
    for n in list(g.node):
        if n.op_type != 'LayerNormalization' or not n.name.endswith('/norm3/LayerNormalization'):
            continue
        tb = n.name[:-len('norm3/LayerNormalization')]
        k = FF_CHUNKS[level_of(tb)]
        if k <= 1:
            continue
        x, y = n.output[0], tb + 'ff/net.2/Add_output_0'
        region = [m for m in g.node if m.name.startswith(tb + 'ff/') and m.op_type != 'Constant']
        made = {o for m in region for o in m.output}
        if y not in made:
            raise SystemExit(f'{tb}: 全結合の出力が見つかりません')
        inits = {i.name for i in g.initializer}
        consts = {o for m in g.node if m.op_type == 'Constant' for o in m.output}
        for m in region:   # 外からの入力は x・重み・定数だけであること
            for i in m.input:
                if i and i not in made and i != x and i not in inits and i not in consts:
                    raise SystemExit(f'{tb}: {m.name} が想定外の入力 {i} を使っています')
        for o in made:     # 外で使われるのは y だけであること
            if o != y and (o in outputs or any(c not in region for c in consumers.get(o, []))):
                raise SystemExit(f'{tb}: {o} が全結合の外でも使われています')

        def make(nodes, tb=tb, x=x, y=y, region=region, made=made, k=k):
            b = tb + 'ff/chunk'
            parts = []
            for i, xi in enumerate(chunk_bounds(g, nodes, x, 1, k, b)):
                ren = lambda t: xi if t == x else (f'{t}/c{i}' if t in made else t)
                for m in region:
                    c = onnx.NodeProto()
                    c.CopyFrom(m)
                    c.name = f'{m.name}/c{i}'
                    del c.input[:]
                    c.input.extend(ren(t) for t in m.input)
                    del c.output[:]
                    c.output.extend(ren(t) for t in m.output)
                    nodes.append(c)
                parts.append(f'{y}/c{i}')
            concat_tree(nodes, parts, y, b, 1)
        last = max(region, key=lambda m: list(g.node).index(m))
        replace[id(last)] = make
        drop.update(id(m) for m in region)
        gone.update(made)
        done.append(('全結合', tb, k))

    new_nodes = []
    for n in g.node:
        if id(n) in replace:
            replace[id(n)](new_nodes)
        elif id(n) not in drop:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    vis = [v for v in g.value_info if v.name not in gone]
    del g.value_info[:]
    g.value_info.extend(vis)
    return done


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src, load_external_data=False)
    done = slice_unet(m)
    onnx.save(m, dst)
    short = lambda p: p.strip('/').split('/transformer')[0].replace('/attentions.', '.').rstrip('/')
    for kind in ('自己注意', '全結合'):
        items = [f'{short(p)}×{k}' for (t, p, k) in done if t == kind]
        print(f'{dst}: {kind} {len(items)}か所を小分け（{", ".join(items)}）')
