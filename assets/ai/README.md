# assets/ai — 端末内AI超解像

| ファイル | 由来 | ライセンス |
| --- | --- | --- |
| `tf.min.js` | @tensorflow/tfjs 4.22.0 `dist/tf.es2017.min.js` | Apache-2.0 |
| `tf-backend-webgpu.min.js` | @tensorflow/tfjs-backend-webgpu 4.22.0 | Apache-2.0 |
| `upscaler.min.js` | @plotdb/upscaler 0.0.5（タイル分割・継ぎ目ブレンド・IndexedDBキャッシュ） | ISC |
| `models/realcugan/2x-conservative-64` | Real-CUGAN up2x-latest-conservative（bilibili） | MIT（`models/realcugan/LICENSE`） |
| `models/realesrgan/anime_fast-64` | Real-ESRGAN realesr-animevideov3（Xintao Wang） | BSD-3-Clause（`models/realesrgan/LICENSE`） |
| `models/realesrgan/general_fast-64` | Real-ESRGAN realesr-general-x4v3 + wdn-x4v3 | BSD-3-Clause（`models/realesrgan/LICENSE`） |

モデルは公式 PyTorch 重みを TF.js（float16, 64×64 タイル）に変換したもので、@plotdb/upscaler の npm パッケージに同梱されているものをそのまま使っています。

## 端末内（オフライン）画像生成

| ファイル | 由来 | ライセンス |
| --- | --- | --- |
| `ort/ort.webgpu.min.js` | onnxruntime-web 1.31.1（WebGPU 版） | MIT |
| `ort/ort-wasm-simd-threaded.asyncify.{mjs,wasm}` | onnxruntime-web 1.31.1 の実行部（`.mjs` に「LGO3」と書いた小さな修正あり。下記） | MIT |
| `local-gen-core.js` | CLIP トークナイザ／DPM-Solver++ 2M／生成パイプライン（本アプリ独自） | — |
| `local-gen.js` | モデルの取得・保存・実行の管理（本アプリ独自） | — |
| `local-gen-worker.js` | OPFS への保存・再開・SHA-256 検証を行う Worker（本アプリ独自） | — |
| `local-gen-run-worker.js` | 生成を Worker で実行。重みを1個ずつ読んで GPU へ送る（本アプリ独自） | — |
| `graphs/{M,S}/*.onnx` | 各モデルの「設計図」（重みなし・計2.6MB）。`tools/make_external_graph.py` で生成し、文章部品は `tools/prune_graph.py` で使わない要約出力を除去、描画部品は `tools/slice_unet.py` で大きな計算を小分け | 元モデルに従う |

### 設計図（外部データ方式）について
モデル丸ごと（例: unet 647MB）を onnxruntime-web に渡すと、wasm メモリへ複製されて約2.4倍に膨らみ、
iPhone（Safari のページ上限 約1.9GB）で強制終了していました。設計図＋元ファイルを `externalData` として渡すと、
重みは wasm を通らず GPU へ直接送られます。

| 計測（Chromium・WebGPU・fp32） | 旧方式 | 設計図方式 |
| --- | --- | --- |
| unet（1.23GB）作成後の wasm メモリ | 2967 MB | 16 MB |
| vae_decoder（0.19GB） | 412 MB | 16 MB |
| 出力 | — | 完全一致 |

設計図の重みは「元の model.onnx 内のバイト位置」を指すため、配布元のファイルを未改変のまま使えます。

モデル本体はリポジトリに含めず、利用者の端末が初回だけ Hugging Face から取得して OPFS に保存します。

| 種類 | 取得元（リビジョン固定 `fa62aea`） | 容量 | ライセンス |
| --- | --- | --- | --- |
| 標準（fp16） | `cursedhelm/aderpy-deepdreamer-onnx` の `tiny-sd-web-fp16` | 約 0.95 GB | CreativeML Open RAIL-M |
| 軽量（4bit） | 同 `tiny-sd-web-q4f16` | 約 0.60 GB | CreativeML Open RAIL-M |

元モデルは [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd)。CreativeML Open RAIL-M は個人利用を認める一方、
違法・有害な用途などを禁止しています。

検証:
- トークナイザ … Hugging Face `tokenizers` と 6/6 一致（日本語・記号・特殊トークン含む）
- スケジューラ … diffusers `DPMSolverMultistepScheduler` と誤差 3e-6 以内で一致（4〜40工程）
- ブラウザ（onnxruntime-web）と Node（onnxruntime-node）の出力 … 平均差 1/255

### iPhone の強制終了への対策（実機テスト第1〜8回の結論）
当時は iPhone の Safari に「JS のメモリ＋GPU のメモリ」の合計約1.9GBの上限があると推定していた（第9回で、主な原因は下記の3秒ルールと判明）。描画の作業メモリは
256×384 で約80MB・448×640 でも約625MB（グラフから計算）と小さく、問題は「モデルファイル（444〜647MB）を
丸ごとメモリへ読んだ写しが、GPU へ送った後も残ること」だった。

| 対策 | 内容 |
| --- | --- |
| Worker で実行 | 1枚ごとに Worker を作って捨てる → GPU を含むメモリが確実に返る |
| 重みを1個ずつ読む | OPFS の同期ハンドルで必要な分だけ読み、作業用の1枚を使い回す（最大56MB） |
| `.mjs` の修正（LGO3） | 外部データの読み出しを `readSlice` に対応／転送方式を切替可能に（既定は元の方式） |
| GPU バッファ | 使い回し（bucket）をやめ、使用後すぐ解放（lazyRelease） |
| GPU 接続の「つなぎ」 | onnxruntime-web 1.31 は最後のセッションを解放すると GPU 接続を破棄し、次の部品で作り直す。iPhone では文章部品→描画部品の切り替え直後の描画が毎回（通算10回）強制終了していたため、生成の間は極小モデル（x+x）のセッションを持ち続けて接続を保つ |
| 文章部品の要約出力を除去 | `pooler_output` はアプリで使わないが、ArgMax（最大値の位置）→その位置での読み出しという、データ次第の計算をしていた。`last_hidden_state` だけを残す（WebGPU 半精度で完全一致を確認） |
| 部品の URL に版 | Worker が読み込む部品だけ古い保存データが使われることがあったため、`?v=` を付ける（`local-gen.js` の `AV`、`index.html`、`diag-gen.html`、`sw.js` をそろえる） |

出力は修正前と完全一致（Chromium・WebGPU／CPU とも）。

### 大きい画像の作業メモリを減らす（実機テスト第8回の結論）
第8回で、描画部品だけを動かしても 192×256 以下は毎回成功・256×384 以上はほぼ毎回強制終了と分かった。
大きさで急に増えるのは次の2つで、どちらも「画素の並び」で小分けにしても計算結果は変わらない。

| 計算 | 448×640 での中間結果 | 小分け（`tools/slice_unet.py`） |
| --- | --- | --- |
| 自己注意（画素どうしの関係の表、画素数×画素数） | 1個 306MB | 問い合わせ側を最大解像度で32分割・1/2解像度で2分割 → 10MB 以下 |
| 全結合（1画素ずつ 320→2560→320 に拡大・縮小） | 1個 22MB | 入力を4分割（1/2解像度は2分割）→ 6MB 以下 |

| 計測（Dawn＋lavapipe・軽量モデル・GPU バッファの同時最大） | 192×256 | 256×384 | 448×640 |
| --- | --- | --- | --- |
| 元の設計図 | 501 MB | 596 MB | 1316 MB（出力が全部 NaN） |
| 小分け版 | — | 502 MB | 638 MB |

- 出力は元と完全一致（256×384・448×640 の軽量、128×128 の標準）。448×640 の元の版が NaN になるのは、表1個が
  「1回に結び付けられるバッファの上限」（Dawn は128MB。iPhone は1GB）を超えるため。Android 等でも起こりうる
- 1工程の GPU 命令数は 757 → 1303 に増える（lavapipe では所要時間ほぼ同じ）
- 分割数は `ATTN_CHUNKS` / `FF_CHUNKS`。分割位置は `Shape` から計算し、大きさが決まると onnxruntime が定数に畳み込む

### 強制終了の本当の原因: Safari の「3秒ルール」（実機テスト第9回の結論）
第9回で、GPU メモリは計算なしなら 1.5GB まで確保できた（メモリ量は原因ではない）。強制終了の記録
（解析データの `com.apple.WebKit.GPU-日付.ips`）は、`WebGPU::crashGPUProcess` ← `Queue::commitMTLCommandBuffer`
のタイマーだった。WebKit（`Source/WebGPU/WebGPU/Queue.mm`）は、GPU へ送った命令のまとまり（command buffer）が
**送ってから3秒以内に終わらないと GPU プロセスごと落とす**。onnxruntime は命令を一気に送るため、
大きい画像では後ろの命令が前の計算待ちで3秒を超えていた（448×640 は描画18工程を完走し、仕上げで落ちた）。

| 対策（`local-gen-run-worker.js` の `installPace`、`local-gen.js` の `pace`） | 内容 |
| --- | --- |
| 1回に送る命令を2個に | セッション設定 `ep.webgpuexecutionprovider.maxNumPendingDispatches`（既定16）。GPU 接続ごとに最初のセッションの値が効くので、つなぎ用セッションにも付ける |
| GPU に預ける量を抑える | 最近16回の実行時間の最大から、同時に預ける送信の数（2〜16）を決め、見積もり合計0.8秒以内にする。残りは手元の列で待たせ、終わった分だけ送る |
| 順序を保つ | 列が空でない間は `writeBuffer`（中身を写して）・バッファ破棄・`mapAsync`・`onSubmittedWorkDone` も同じ列に並べる |

Dawn＋lavapipe（軽量・128×128）で、送信の「待ち」の最大は 9.06秒 → 0.80秒、出力は完全一致、所要時間はほぼ同じ。

#### 半精度 WebGPU の再現環境（開発用）
Chromium 同梱の SwiftShader は半精度(f16)に非対応のため、Node.js ＋ Dawn（npm `webgpu`）＋ Mesa lavapipe
（`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`）で、実際の軽量・標準モデルを WebGPU 半精度で通し生成できる。
ここでは通し生成がエラーなく完了し、計算プログラム55個・GPU メモリ最大540MB（128×128）だった。
