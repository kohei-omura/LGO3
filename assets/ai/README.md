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
| `ort/ort-wasm-simd-threaded.asyncify.{mjs,wasm}` | onnxruntime-web 1.31.1 の実行部 | MIT |
| `local-gen-core.js` | CLIP トークナイザ／DPM-Solver++ 2M／生成パイプライン（本アプリ独自） | — |
| `local-gen.js` | モデルの取得・保存・実行の管理（本アプリ独自） | — |
| `local-gen-worker.js` | OPFS への保存・再開・SHA-256 検証を行う Worker（本アプリ独自） | — |

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
