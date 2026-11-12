#!/usr/bin/env python3
"""
設計図(graph)から、使わない出力とそれを作るためだけの計算を取り除く。

例: 文章部品(text_encoder)の pooler_output はアプリで使わないが、作るために
    ArgMax（最大値の位置）→ その位置での読み出し、というデータ次第の計算をしている。
    iPhone の Safari で強制終了の疑いがあるため、出力ごと取り除く。

重み(initializer)は外部データ（元ファイル内のバイト位置）を指したまま。使われなくなった
重みは設計図から外すだけで、元ファイルはそのまま使える。

使い方: prune_graph.py <入力graph.onnx> <出力graph.onnx> <残す出力名> [<残す出力名> ...]
"""
import sys
import onnx


def prune(model, keep):
    g = model.graph
    names = [o.name for o in g.output]
    for k in keep:
        if k not in names:
            raise SystemExit(f'出力 {k} がありません（あるのは {names}）')
    # 残す出力から逆向きにたどり、必要な計算だけを残す
    producer = {o: n for n in g.node for o in n.output}
    need, stack = set(), list(keep)
    while stack:
        t = stack.pop()
        if t in need:
            continue
        need.add(t)
        n = producer.get(t)
        if n is not None:
            stack.extend(i for i in n.input if i)
    used_nodes = [n for n in g.node if any(o in need for o in n.output)]
    removed = len(g.node) - len(used_nodes)
    removed_ops = sorted({n.op_type for n in g.node if not any(o in need for o in n.output)})
    del g.node[:]
    g.node.extend(used_nodes)
    outs = [o for o in g.output if o.name in keep]
    del g.output[:]
    g.output.extend(outs)
    inits = [i for i in g.initializer if i.name in need]
    n_init = len(g.initializer) - len(inits)
    del g.initializer[:]
    g.initializer.extend(inits)
    vis = [v for v in g.value_info if v.name in need]
    del g.value_info[:]
    g.value_info.extend(vis)
    return removed, removed_ops, n_init


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    src, dst, keep = sys.argv[1], sys.argv[2], sys.argv[3:]
    m = onnx.load(src, load_external_data=False)
    removed, ops, n_init = prune(m, keep)
    onnx.save(m, dst)
    print(f'{dst}: 計算 {removed}個を削除（{", ".join(ops)}）/ 使わない重み {n_init}個を外した')
