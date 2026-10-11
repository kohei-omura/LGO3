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
