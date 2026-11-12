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
| `graphs/{M,S}/*.onnx` | 各モデルの「設計図」（重みなし・計1.9MB）。`tools/make_external_graph.py` で生成 | 元モデルに従う |

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

### iPhone の強制終了への対策（実機テスト第1〜5回の結論）
iPhone の Safari は「JS のメモリ＋GPU のメモリ」の合計に約1.9GBの上限があると推定。描画の作業メモリは
256×384 で約80MB・448×640 でも約625MB（グラフから計算）と小さく、問題は「モデルファイル（444〜647MB）を
丸ごとメモリへ読んだ写しが、GPU へ送った後も残ること」だった。

| 対策 | 内容 |
| --- | --- |
| Worker で実行 | 1枚ごとに Worker を作って捨てる → GPU を含むメモリが確実に返る |
| 重みを1個ずつ読む | OPFS の同期ハンドルで必要な分だけ読み、作業用の1枚を使い回す（最大56MB） |
| `.mjs` の修正（LGO3） | 外部データの読み出しを `readSlice` に対応／転送方式を切替可能に（既定は元の方式） |
| GPU バッファ | 使い回し（bucket）をやめ、使用後すぐ解放（lazyRelease） |

出力は修正前と完全一致（Chromium・WebGPU／CPU とも）。
