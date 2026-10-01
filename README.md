# Suriyos — 3D衝突実験ラボ

物の落下と衝突を、実際に動かして確かめる macOS アプリです。

**[ブラウザで試す](https://suriyos.phyowaisoe.com)**(インストール不要のWebAssembly版)

## 1. 何の役に立つか

物理を、式を読むだけでなく、目で見て確かめられます。

- 球・箱・円柱・円錐・カプセルなどを並べ、ゴム・コンクリート・アルミ・氷・レンガといった材質を選んで、落としたりぶつけたりできます。
- 重力を地球・ほかの惑星・無重力に変えられます。同じ実験が重力でどう変わるかが、その場で分かります。
- ヒンジ・バネ・固定などの拘束で物をつなげられます。力積を加えて弾き飛ばすこともできます。
- 一時停止とコマ送りができるので、衝突の瞬間を止めて観察できます。
- 実験の設定と時系列データを JSON / CSV に書き出せるので、レポートや分析にそのまま使えます。
- 日本語と英語の画面。授業でも自習でも使えます。

## 2. 使っている技術

- **C++** — アプリ本体(`src/main.cpp`)
- **Bullet Physics** — 剛体の衝突、拘束、落下の計算
- **Dear ImGui + GLFW + OpenGL 3** — 画面と3D描画
- **Emscripten(WebAssembly)** — `build_web.sh` でブラウザ版をビルド
- **Makefile** — `make` / `make run` / `make app`(アイコン付きの .app にまとめる)

必要環境とセットアップ:

```sh
brew install glfw bullet
git clone --depth 1 https://github.com/ocornut/imgui.git third_party/imgui
make run
```

macOS 11以降(Apple Silicon / Intel)、Xcode Command Line Tools が必要です。
