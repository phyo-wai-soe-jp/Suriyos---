# Suriyos

[Dear ImGui](https://github.com/ocornut/imgui)、GLFW、[Bullet Physics](https://pybullet.org/wordpress/index.php/forum-2/) を使用した macOS 向け3D衝突・落下物理シミュレーター。日本語/英語UIに対応し、シミュレーション結果を分析用にエクスポートできます。

## 必要環境

- macOS 11以降(Apple Silicon / Intel)
- Xcode Command Line Tools(`xcode-select --install`)
- [Homebrew](https://brew.sh)

## セットアップ

```sh
brew install glfw bullet
git clone --depth 1 https://github.com/ocornut/imgui.git third_party/imgui
```

## ビルド・実行

```sh
make        # Suriyos バイナリをビルド
make run    # ビルドして起動
make app    # アイコン付きの Suriyos.app にまとめる
```

## プロジェクト構成

```
Suriyos/
├── src/main.cpp        # アプリケーション本体のソースコード
├── Resources/           # `make app` で使用する Info.plist とアイコン
└── Makefile
```
