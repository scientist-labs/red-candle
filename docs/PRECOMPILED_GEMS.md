# Precompiled Gems

Red Candle is a Ruby gem with a large Rust extension. By default, installing it
compiles that extension on the user's machine, which requires the Rust toolchain
(`rustc`/`cargo`) and several minutes of build time. To remove that friction we
publish **precompiled platform gems** for common platforms; users on those
platforms get a ready-to-run binary and need no Rust toolchain at all.

## What ships

| Gem platform | Acceleration | Toolchain needed to install |
|---|---|---|
| `x86_64-linux` | CPU | none (precompiled) |
| `aarch64-linux` | CPU | none (precompiled) |
| `x64-mingw-ucrt` (Windows) | CPU | none (precompiled) |
| `ruby` (source) | auto-detected: **Metal** (macOS), **CUDA**, MKL, Accelerate, or CPU | Rust toolchain |

RubyGems automatically selects the precompiled gem matching the user's
`arch-os` + Ruby ABI and **falls back to the source gem** when there is no
match. So `gem install red-candle` "just works" everywhere; only platforms
without a binary compile from source.

## Why macOS/Metal and CUDA stay source-only

The two GPU backends have opposite characteristics, and only one is hard to
precompile:

- **Metal (macOS)** is a system framework on every Mac — no driver/SDK
  variability — so in principle a darwin binary could bake it in. The blocker is
  the **build**: `candle-metal-kernels` compiles Metal shaders with
  `xcrun metal` at build time, which requires a real macOS runner with Xcode.
  The Linux cross-compile containers (`rake-compiler-dock`) can't do this, so
  darwin+Metal precompilation is deferred to a future phase (it needs dedicated
  macOS runners plus multi-ABI cross-rubies). Until then, macOS users build from
  the source gem — they already have Xcode, and the Metal build is quick.
- **CUDA** cannot ship as a single universal binary at all: it depends on the
  NVIDIA driver + a specific CUDA runtime (11.x vs 12.x), targets specific GPU
  compute capabilities, and requires `nvcc` to compile candle's kernels. Baking
  CUDA into the default Linux gem would also break the ~95% of machines with no
  NVIDIA GPU. CUDA users therefore build from source, which is a small ask since
  they already need `nvcc` locally.

### Forcing the source gem (CUDA / custom acceleration)

```ruby
# Gemfile
gem "red-candle", force_ruby_platform: true
```

```bash
gem install red-candle --platform=ruby
```

`extconf.rb` then auto-detects CUDA/Metal/MKL from the local environment (and
respects `CANDLE_FEATURES`, `CANDLE_FORCE_CPU`, `CANDLE_DISABLE_CUDA`).

## How it's built (CI)

Building happens in `.github/workflows/release.yml`, triggered manually via
**workflow_dispatch**. The pipeline:

1. **prepare** — bump `version.rb`, commit, tag (skipped on dry runs).
2. **cross-gems** — build each Linux/Windows platform gem in a
   `rake-compiler-dock` container via
   [`oxidized-rb/cross-gem-action`](https://github.com/oxidized-rb/cross-gem-action),
   packing one native extension per Ruby ABI (3.1–3.4). The container has no
   CUDA and isn't macOS, so the build is CPU-only (`CANDLE_FORCE_CPU=1` makes
   that explicit).
3. **source-gem** — `gem build` the portable fallback gem.
4. **smoke-test** — install each precompiled gem on a stock runner **with no
   Rust toolchain** and `require "candle"`. This is the real proof the binary
   is self-contained.
5. **publish** — push every gem (platform gems + source) to RubyGems and create
   a GitHub release (skipped on dry runs).

### Dry runs

Run the workflow with **`dry_run: true`** to exercise the entire build +
smoke-test matrix and upload the resulting gems as run artifacts, **without**
bumping the version, tagging, or publishing. Use this to validate the pipeline
(and inspect/`gem install` the artifacts locally) before a real release.

### The Rakefile hook

`Rakefile` uses `RbSys::ExtensionTask` (a `rake-compiler` subclass) with
`cross_compile = true`, which registers the `native` / `native:<platform>`
tasks that `cross-gem-action` invokes inside the dock.

## Applying this pattern to other Rust gems

Most Rust-backed Ruby gems in this workspace are simpler than Red Candle (no
Metal/CUDA), so the **full set of platforms — including macOS — can be
precompiled via `cross-gem-action`** with no special-casing. To replicate:

1. **Depend on `rb_sys`** and use `rb_sys/mkmf`'s `create_rust_makefile` in
   `extconf.rb` (most rb-sys gems already do).
2. **Rakefile**: use `RbSys::ExtensionTask` with `cross_compile = true` and a
   `cross_platform` list. Add macOS platforms (`arm64-darwin`, `x86_64-darwin`)
   too when the gem has no Metal/Xcode build step.
3. **release.yml**: copy this workflow. For a pure-CPU gem, add the darwin
   platforms to the `cross-gems` matrix and drop the source-only macOS caveat;
   you can also drop `CANDLE_FORCE_CPU`.
4. Keep the **source gem + `dry_run` + clean-room smoke test** — they're the
   cheap insurance that makes the whole thing trustworthy.

The only red-candle-specific complication is the `xcrun metal` shader build; a
gem without it has no such constraint.
