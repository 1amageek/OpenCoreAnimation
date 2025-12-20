# 不完全実装の設計書

このドキュメントでは、OpenCoreAnimationの未実装機能について、Apple CoreAnimationのAPI仕様に準拠した実装設計を記述します。

---

## 1. CAEmitterLayer - パーティクルシステム

### 現状

- `CAEmitterLayer.swift`: プロパティのみ定義（55行）
- `CAEmitterCell.swift`: プロパティとCAMediaTiming準拠のみ（195行）
- レンダラーにパーティクル処理なし

### Apple仕様

```swift
class CAEmitterLayer: CALayer {
    var emitterCells: [CAEmitterCell]?
    var emitterPosition: CGPoint      // エミッター中心位置
    var emitterZPosition: CGFloat     // Z軸位置
    var emitterSize: CGSize           // エミッターサイズ
    var emitterDepth: CGFloat         // エミッター深度
    var emitterShape: CAEmitterLayerEmitterShape  // point, line, rectangle, circle, sphere, etc.
    var emitterMode: CAEmitterLayerEmitterMode    // points, outline, surface, volume
    var renderMode: CAEmitterLayerRenderMode      // unordered, oldestFirst, oldestLast, backToFront, additive
    var preservesDepth: Bool
    var birthRate: Float              // セルのbirthRateに乗算
    var lifetime: Float               // セルのlifetimeに乗算
    var velocity: Float               // セルのvelocityに乗算
    var scale: Float                  // セルのscaleに乗算
    var spin: Float                   // セルのspinに乗算
    var seed: UInt32                  // 乱数シード
}

class CAEmitterCell: CAMediaTiming {
    var contents: Any?                // CGImage
    var contentsRect: CGRect
    var birthRate: Float              // パーティクル/秒
    var lifetime: Float               // 秒
    var lifetimeRange: Float
    var velocity: CGFloat
    var velocityRange: CGFloat
    var xAcceleration, yAcceleration, zAcceleration: CGFloat
    var scale: CGFloat
    var scaleRange: CGFloat
    var scaleSpeed: CGFloat
    var spin: CGFloat                 // rad/sec
    var spinRange: CGFloat
    var emissionLatitude: CGFloat     // 緯度角
    var emissionLongitude: CGFloat    // 経度角
    var emissionRange: CGFloat        // 放射角度範囲
    var color: CGColor?
    var redRange, greenRange, blueRange, alphaRange: Float
    var redSpeed, greenSpeed, blueSpeed, alphaSpeed: Float
    var emitterCells: [CAEmitterCell]?  // サブエミッター
    var name: String?
    var isEnabled: Bool
}
```

### 実装設計

#### 1.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    CAEmitterLayer                            │
│  (プロパティ定義 + シミュレーション管理)                       │
├─────────────────────────────────────────────────────────────┤
│                  ParticleSimulator                           │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ ParticlePool    │  │ EmissionController│                  │
│  │ (メモリ管理)    │  │ (生成タイミング)  │                  │
│  └─────────────────┘  └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                 CAWebGPURenderer拡張                         │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ ParticleBuffer  │  │ ParticleShader   │                  │
│  │ (GPU頂点バッファ)│  │ (インスタンス描画)│                  │
│  └─────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2 パーティクルデータ構造

```swift
/// 単一パーティクルの状態
struct Particle {
    var position: SIMD3<Float>      // 現在位置
    var velocity: SIMD3<Float>      // 速度
    var acceleration: SIMD3<Float>  // 加速度
    var color: SIMD4<Float>         // 現在色
    var colorSpeed: SIMD4<Float>    // 色変化速度
    var scale: Float                // 現在スケール
    var scaleSpeed: Float           // スケール変化速度
    var spin: Float                 // 現在回転角
    var spinSpeed: Float            // 回転速度
    var lifetime: Float             // 残り寿命
    var maxLifetime: Float          // 初期寿命
    var textureIndex: UInt16        // テクスチャ配列インデックス
    var isAlive: Bool
}

/// パーティクルプール（オブジェクトプール）
final class ParticlePool {
    private var particles: [Particle]
    private var aliveCount: Int = 0
    private let maxParticles: Int

    func spawn() -> Int?  // 空きスロットのインデックスを返す
    func kill(at index: Int)
    func update(deltaTime: Float)
}
```

#### 1.3 エミッション形状

```swift
/// エミッション位置を計算
func emissionPosition(
    shape: CAEmitterLayerEmitterShape,
    mode: CAEmitterLayerEmitterMode,
    position: CGPoint,
    zPosition: CGFloat,
    size: CGSize,
    depth: CGFloat,
    seed: inout UInt32
) -> SIMD3<Float> {
    switch shape {
    case .point:
        return SIMD3(Float(position.x), Float(position.y), Float(zPosition))
    case .line:
        let t = randomFloat(&seed)
        return SIMD3(
            Float(position.x - size.width/2 + size.width * CGFloat(t)),
            Float(position.y),
            Float(zPosition)
        )
    case .rectangle:
        // mode: outline → 辺上, surface → 面上, volume → 面上
        ...
    case .circle:
        // mode: outline → 円周上, surface/volume → 円内
        let angle = randomFloat(&seed) * 2 * .pi
        let r = mode == .outline ? 1.0 : sqrt(randomFloat(&seed))
        ...
    case .sphere:
        // 球面または球内
        ...
    case .cuboid:
        // 直方体
        ...
    }
}
```

#### 1.4 GPU描画（インスタンス描画）

```wgsl
// パーティクルシェーダー
struct ParticleInstance {
    @location(0) position: vec3f,
    @location(1) color: vec4f,
    @location(2) scale_rotation: vec2f,  // x: scale, y: rotation
}

@vertex
fn vs_particle(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
    instance: ParticleInstance
) -> VertexOutput {
    // ビルボード用クアッド頂点
    let corners = array<vec2f, 6>(
        vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(-0.5, 0.5),
        vec2f(0.5, -0.5), vec2f(0.5, 0.5), vec2f(-0.5, 0.5)
    );

    var corner = corners[vertexIndex];

    // 回転適用
    let cos_r = cos(instance.scale_rotation.y);
    let sin_r = sin(instance.scale_rotation.y);
    corner = vec2f(
        corner.x * cos_r - corner.y * sin_r,
        corner.x * sin_r + corner.y * cos_r
    );

    // スケール適用
    corner *= instance.scale_rotation.x;

    var output: VertexOutput;
    output.position = uniforms.mvp * vec4f(instance.position + vec3f(corner, 0.0), 1.0);
    output.color = instance.color;
    output.texCoord = corners[vertexIndex] + vec2f(0.5);
    return output;
}

@fragment
fn fs_particle(input: VertexOutput) -> @location(0) vec4f {
    let texColor = textureSample(particleTexture, particleSampler, input.texCoord);
    return texColor * input.color;
}
```

#### 1.5 レンダーモード実装

| renderMode | 実装 |
|------------|------|
| `unordered` | インスタンスバッファ順で描画 |
| `oldestFirst` | 寿命で昇順ソート |
| `oldestLast` | 寿命で降順ソート |
| `backToFront` | Z値で降順ソート |
| `additive` | ブレンドモードを`src + dst`に変更 |

#### 1.6 実装ファイル構成

```
Sources/OpenCoreAnimation/
├── Emitter/
│   ├── ParticleSimulator.swift      # シミュレーションロジック
│   ├── ParticlePool.swift           # メモリ管理
│   ├── EmissionShape.swift          # 形状計算
│   └── ParticleShaders.swift        # WGSLシェーダー
├── CAEmitterLayer.swift             # 既存（変更なし）
└── CAEmitterCell.swift              # 既存（変更なし）
```

---

## 2. CATiledLayer - タイル描画

### 現状

- `CATiledLayer.swift`: 3プロパティと`fadeDuration()`のみ（19行）
- タイル分割・非同期読み込み・LOD管理なし

### Apple仕様

```swift
class CATiledLayer: CALayer {
    var levelsOfDetail: Int       // 縮小時のLODレベル数（デフォルト1）
    var levelsOfDetailBias: Int   // 拡大時のLODレベル数（デフォルト0）
    var tileSize: CGSize          // タイルサイズ（デフォルト256x256）

    class func fadeDuration() -> CFTimeInterval  // フェードイン時間（0.25秒）
}
```

**動作仕様:**
- `draw(in:)` がバックグラウンドスレッドで呼ばれる
- CTMからタイルの位置と解像度を判断
- `setNeedsDisplay(_:)` で特定領域を無効化

### 実装設計

#### 2.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    CATiledLayer                              │
│  (タイル座標計算 + 描画リクエスト管理)                         │
├─────────────────────────────────────────────────────────────┤
│                   TileManager                                │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ TileCache       │  │ TileRenderer     │                  │
│  │ (LRUキャッシュ) │  │ (非同期描画)     │                  │
│  └─────────────────┘  └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│              CAWebGPURenderer拡張                            │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │ TileTexture     │  │ FadeAnimation    │                  │
│  │ (GPUテクスチャ) │  │ (フェードイン)   │                  │
│  └─────────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2 タイル座標システム

```swift
/// タイルの識別子
struct TileKey: Hashable {
    let x: Int           // タイルX座標
    let y: Int           // タイルY座標
    let lod: Int         // LODレベル（0が最高解像度）
}

/// LODレベル計算
func calculateLOD(scale: CGFloat, levelsOfDetail: Int, levelsOfDetailBias: Int) -> Int {
    // scale < 1: 縮小 → 正のLODレベル
    // scale > 1: 拡大 → 負のLODレベル（bias内）
    let logScale = log2(Double(scale))
    let lod = -Int(floor(logScale))
    return max(-levelsOfDetailBias, min(levelsOfDetail - 1, lod))
}

/// 可視タイルの列挙
func visibleTiles(
    visibleRect: CGRect,
    tileSize: CGSize,
    lod: Int
) -> [TileKey] {
    let scaleFactor = pow(2.0, Double(lod))
    let effectiveTileSize = CGSize(
        width: tileSize.width * CGFloat(scaleFactor),
        height: tileSize.height * CGFloat(scaleFactor)
    )

    let minX = Int(floor(visibleRect.minX / effectiveTileSize.width))
    let maxX = Int(ceil(visibleRect.maxX / effectiveTileSize.width))
    let minY = Int(floor(visibleRect.minY / effectiveTileSize.height))
    let maxY = Int(ceil(visibleRect.maxY / effectiveTileSize.height))

    var tiles: [TileKey] = []
    for y in minY..<maxY {
        for x in minX..<maxX {
            tiles.append(TileKey(x: x, y: y, lod: lod))
        }
    }
    return tiles
}
```

#### 2.3 非同期描画（WASM対応）

WASMはシングルスレッドのため、`setTimeout`を使用した協調的マルチタスク:

```swift
#if arch(wasm32)
import JavaScriptKit

final class TileRenderer {
    private var pendingTiles: [TileKey] = []
    private var rendering: Bool = false

    func requestTile(_ key: TileKey, layer: CATiledLayer) {
        pendingTiles.append(key)
        scheduleRender()
    }

    private func scheduleRender() {
        guard !rendering, !pendingTiles.isEmpty else { return }
        rendering = true

        // 次のイベントループで描画
        let callback = JSClosure { [weak self] _ in
            self?.renderNextTile()
            return .undefined
        }
        _ = JSObject.global.setTimeout!(callback, 0)
    }

    private func renderNextTile() {
        guard let key = pendingTiles.first else {
            rendering = false
            return
        }
        pendingTiles.removeFirst()

        // CGContextを作成してdraw(in:)を呼び出す
        let context = createTileContext(for: key)
        layer.draw(in: context)

        // テクスチャに変換してキャッシュ
        let texture = createTexture(from: context)
        tileCache[key] = TileCacheEntry(texture: texture, opacity: 0)

        // 次のタイルをスケジュール
        scheduleRender()
    }
}
#endif
```

#### 2.4 フェードインアニメーション

```swift
struct TileCacheEntry {
    let texture: GPUTexture
    var opacity: Float           // 0→1でフェードイン
    let creationTime: CFTimeInterval

    func currentOpacity(fadeDuration: CFTimeInterval) -> Float {
        let elapsed = CACurrentMediaTime() - creationTime
        return Float(min(1.0, elapsed / fadeDuration))
    }
}
```

#### 2.5 実装ファイル構成

```
Sources/OpenCoreAnimation/
├── Tiling/
│   ├── TileManager.swift          # タイル管理
│   ├── TileCache.swift            # LRUキャッシュ
│   ├── TileRenderer.swift         # 非同期描画
│   └── TileCoordinate.swift       # 座標計算
└── CATiledLayer.swift             # 既存に機能追加
```

---

## 3. CATransformLayer - 3D階層レンダリング

### 現状

- `CATransformLayer.swift`: `hitTest`オーバーライドのみ（13行）
- レンダラーで特別な3D処理なし

### Apple仕様

CATransformLayerは**サブレイヤーをZ=0平面にフラット化しない**。

**無視されるプロパティ:**
- `backgroundColor`, `contents`, `borderWidth`, `borderColor`
- `filters`, `backgroundFilters`, `compositingFilter`
- `mask`, `masksToBounds`
- `shadowColor`, `shadowOffset`, `shadowRadius`, `shadowOpacity`, `shadowPath`

**特殊な動作:**
- `opacity`は各サブレイヤーに個別適用（コンポジットグループを形成しない）
- `hitTest(_:)`は常にnilを返す

### 実装設計

#### 3.1 レンダリング方式

通常のCALayerは2Dコンポジットを行うが、CATransformLayerは真の3Dレンダリング:

```swift
// CAWebGPURenderer内

func renderLayer(_ layer: CALayer, ...) {
    // CATransformLayerの場合、自身のプロパティは描画しない
    if layer is CATransformLayer {
        renderTransformLayerSublayers(layer, ...)
        return
    }

    // 通常のレイヤー描画...
}

private func renderTransformLayerSublayers(
    _ transformLayer: CALayer,
    renderPass: GPURenderPassEncoder,
    parentMatrix: Matrix4x4
) {
    guard let sublayers = transformLayer.sublayers else { return }

    let presentationLayer = transformLayer.presentation() ?? transformLayer
    var sublayerMatrix = presentationLayer.modelMatrix(parentMatrix: parentMatrix)

    if !CATransform3DIsIdentity(presentationLayer.sublayerTransform) {
        sublayerMatrix = sublayerMatrix * presentationLayer.sublayerTransform.matrix4x4
    }

    // 深度テストを有効にして3D描画
    // 各サブレイヤーのzPositionを実際のZ座標として使用
    for sublayer in sublayers.sorted(by: { $0.zPosition < $1.zPosition }) {
        renderLayer(sublayer, renderPass: renderPass, parentMatrix: sublayerMatrix)
    }
}
```

#### 3.2 深度バッファの使用

```swift
// パイプライン設定
let depthStencilState = GPUDepthStencilState(
    format: .depth24plus,
    depthWriteEnabled: true,
    depthCompare: .less
)

// レンダーパス
let renderPassDescriptor = GPURenderPassDescriptor(
    colorAttachments: [...],
    depthStencilAttachment: GPURenderPassDepthStencilAttachment(
        view: depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: .clear,
        depthStoreOp: .store
    )
)
```

#### 3.3 パースペクティブ投影

```swift
/// 3D変換用のパースペクティブ行列
extension Matrix4x4 {
    static func perspective(
        fovY: Float,
        aspect: Float,
        near: Float,
        far: Float
    ) -> Matrix4x4 {
        let y = 1 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)

        return Matrix4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, z * near, 0)
        ))
    }
}
```

---

## 4. シャドウ描画

### 現状

- CALayerにシャドウプロパティあり（`CALayer.swift:1481-1534`）
- CGContextへのシャドウ設定あり（`CALayer.swift:1626-1631`）
- WebGPUレンダラーでシャドウ描画なし

### Apple仕様

```swift
var shadowColor: CGColor?     // デフォルト: opaque black
var shadowOpacity: Float      // デフォルト: 0（0-1）
var shadowOffset: CGSize      // デフォルト: (0, -3)
var shadowRadius: CGFloat     // デフォルト: 3
var shadowPath: CGPath?       // カスタムシャドウ形状
```

### 実装設計

#### 4.1 シャドウレンダリングパイプライン

```
1. シャドウマスク生成
   ├── shadowPath使用時: パスをテクスチャに描画
   └── shadowPath未設定時: レイヤー形状（角丸含む）をテクスチャに描画

2. ガウシアンブラー適用
   ├── 水平パス
   └── 垂直パス

3. シャドウ合成
   └── オフセット適用してメインターゲットに描画
```

#### 4.2 シャドウマスクシェーダー

```wgsl
// シャドウマスク生成（レイヤー形状）
@fragment
fn fs_shadow_mask(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.texCoord;
    let size = uniforms.layerSize;
    let radius = uniforms.cornerRadius;

    // 角丸矩形のSDF
    let p = abs(uv * size - size * 0.5) - size * 0.5 + radius;
    let d = length(max(p, vec2f(0.0))) - radius;

    // シャドウマスク（内側が1、外側が0）
    let mask = 1.0 - smoothstep(-1.0, 0.0, d);
    return vec4f(mask, mask, mask, mask);
}
```

#### 4.3 ガウシアンブラーシェーダー

```wgsl
// 分離可能ガウシアンブラー（水平パス）
@fragment
fn fs_blur_horizontal(input: VertexOutput) -> @location(0) vec4f {
    let texelSize = 1.0 / vec2f(textureDimensions(inputTexture));
    let blurRadius = uniforms.shadowRadius;

    // 9タップガウシアンカーネル
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var result = textureSample(inputTexture, sampler, input.texCoord) * weights[0];

    for (var i = 1; i < 5; i++) {
        let offset = vec2f(f32(i) * blurRadius * texelSize.x, 0.0);
        result += textureSample(inputTexture, sampler, input.texCoord + offset) * weights[i];
        result += textureSample(inputTexture, sampler, input.texCoord - offset) * weights[i];
    }

    return result;
}
```

#### 4.4 シャドウ合成

```swift
func renderShadow(
    layer: CALayer,
    renderPass: GPURenderPassEncoder,
    modelMatrix: Matrix4x4
) {
    guard layer.shadowOpacity > 0,
          let shadowColor = layer.shadowColor else { return }

    // 1. シャドウマスク生成
    let maskTexture = createShadowMask(layer: layer)

    // 2. ブラー適用
    let blurredTexture = applyGaussianBlur(
        texture: maskTexture,
        radius: layer.shadowRadius
    )

    // 3. オフセット適用して描画
    let shadowMatrix = modelMatrix * Matrix4x4(translation: SIMD3(
        Float(layer.shadowOffset.width),
        Float(-layer.shadowOffset.height),  // CoreAnimationはY反転
        -0.001  // レイヤーの背後に
    ))

    renderTextureWithColor(
        blurredTexture,
        color: shadowColor.withAlphaComponent(CGFloat(layer.shadowOpacity)),
        matrix: shadowMatrix,
        renderPass: renderPass
    )
}
```

---

## 5. マスク機能

### 現状

- `CALayer.mask: CALayer?` プロパティあり（`CALayer.swift:1416`）
- WebGPUレンダラーでマスク処理なし

### Apple仕様

- マスクレイヤーのアルファチャンネルがマスクとして機能
- 不透明ピクセル → コンテンツ表示
- 透明ピクセル → コンテンツ非表示
- マスクレイヤーはsuperlayerを持ってはいけない

### 実装設計

#### 5.1 アルファマスク方式

ステンシルバッファではなく、アルファマスクテクスチャを使用（グラデーションマスク対応）:

```
1. マスクレイヤーをオフスクリーンテクスチャに描画
2. メインレイヤーを描画時、マスクテクスチャをサンプリング
3. マスクのアルファ値でフラグメントを乗算
```

#### 5.2 マスクテクスチャ生成

```swift
func createMaskTexture(maskLayer: CALayer, targetSize: CGSize) -> GPUTexture {
    // マスク用オフスクリーンテクスチャ
    let maskTexture = device.createTexture(GPUTextureDescriptor(
        size: GPUExtent3D(width: UInt32(targetSize.width), height: UInt32(targetSize.height)),
        format: .r8unorm,  // アルファのみ
        usage: [.renderAttachment, .textureBinding]
    ))

    // マスクレイヤーを描画
    let encoder = device.createCommandEncoder()
    let renderPass = encoder.beginRenderPass(/* maskTexture */)

    renderLayer(maskLayer, renderPass: renderPass, parentMatrix: .identity)

    renderPass.end()
    device.queue.submit([encoder.finish()])

    return maskTexture
}
```

#### 5.3 マスク適用シェーダー

```wgsl
@group(1) @binding(0) var maskTexture: texture_2d<f32>;
@group(1) @binding(1) var maskSampler: sampler;

@fragment
fn fs_masked(input: VertexOutput) -> @location(0) vec4f {
    let baseColor = input.color;

    // マスクテクスチャからアルファ取得
    let maskAlpha = textureSample(maskTexture, maskSampler, input.texCoord).r;

    // マスク適用
    return vec4f(baseColor.rgb, baseColor.a * maskAlpha);
}
```

#### 5.4 レンダリングフロー

```swift
func renderLayerWithMask(
    _ layer: CALayer,
    renderPass: GPURenderPassEncoder,
    parentMatrix: Matrix4x4
) {
    guard let maskLayer = layer.mask else {
        // マスクなし - 通常描画
        renderLayerContent(layer, renderPass: renderPass, parentMatrix: parentMatrix)
        return
    }

    // 1. マスクテクスチャ生成
    let maskTexture = createMaskTexture(
        maskLayer: maskLayer,
        targetSize: layer.bounds.size
    )

    // 2. マスク用パイプラインに切り替え
    renderPass.setPipeline(maskedPipeline)

    // 3. マスクテクスチャをバインド
    let maskBindGroup = device.createBindGroup(/* maskTexture */)
    renderPass.setBindGroup(1, bindGroup: maskBindGroup)

    // 4. レイヤーコンテンツ描画
    renderLayerContent(layer, renderPass: renderPass, parentMatrix: parentMatrix)

    // 5. パイプラインを元に戻す
    renderPass.setPipeline(defaultPipeline)
}
```

---

## 実装優先度

| 機能 | 複雑度 | 依存関係 | 優先度 |
|------|--------|---------|--------|
| **シャドウ描画** | 中 | なし | 1 (高) |
| **マスク機能** | 中 | なし | 2 |
| **CATransformLayer** | 低 | 深度バッファ | 3 |
| **CATiledLayer** | 高 | 非同期描画 | 4 |
| **CAEmitterLayer** | 非常に高 | パーティクルシステム全体 | 5 (低) |

### 推奨実装順序

1. **シャドウ描画**: 視覚的インパクトが大きく、比較的シンプル
2. **マスク機能**: シャドウと同様のオフスクリーンレンダリング技術を使用
3. **CATransformLayer**: 深度バッファの追加が必要だが、レンダリングロジックはシンプル
4. **CATiledLayer**: 非同期処理とキャッシュ管理が複雑
5. **CAEmitterLayer**: 完全なパーティクルシステムの実装が必要

---

## テスト計画

### シャドウ描画テスト
```swift
@Test func testShadowRendering() {
    let layer = CALayer()
    layer.frame = CGRect(x: 50, y: 50, width: 100, height: 100)
    layer.backgroundColor = CGColor(red: 1, green: 0, blue: 0, alpha: 1)
    layer.shadowColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
    layer.shadowOffset = CGSize(width: 5, height: 5)
    layer.shadowRadius = 10
    layer.shadowOpacity = 0.5

    // レンダリング結果を検証
    let renderer = CAWebGPURenderer()
    renderer.render(layer: layer)
    // ピクセル検証...
}
```

### マスクテスト
```swift
@Test func testMaskLayer() {
    let layer = CALayer()
    layer.frame = CGRect(x: 0, y: 0, width: 100, height: 100)
    layer.backgroundColor = CGColor(red: 0, green: 0, blue: 1, alpha: 1)

    let maskLayer = CALayer()
    maskLayer.frame = layer.bounds
    maskLayer.cornerRadius = 50
    maskLayer.backgroundColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1)

    layer.mask = maskLayer

    // 円形にマスクされていることを検証
}
```

---

## 参考資料

- [CAEmitterLayer - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/caemitterlayer)
- [CAEmitterCell - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/caemittercell)
- [CATiledLayer - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/catiledlayer)
- [CATransformLayer - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/catransformlayer)
- [CALayer.mask - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/calayer/mask)
- [CALayer Shadow Properties - Apple Developer Documentation](https://developer.apple.com/documentation/quartzcore/calayer/shadowcolor)
