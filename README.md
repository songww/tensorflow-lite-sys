# tensorflow-lite-sys

This crate provides bindings to the raw low-level C API.

## platforms

[✅] With iOS framework(Manual Compilation or CocoaPods).

[] With Android (Manual Compilation or AAR).

[✅] Manual Compilation for macOS.

[✅] Manual Compilation for Linux.

[] Manual Compilation(libtensorflow-lite.a)

## iOS Notice

* static linked by default.
* iOS dose not export experimental apis, enable it should with manual compilation framworks.
* Set env `TFLITE_FRAMEWORK_PATH` for path that contains `TensorFlowLiteC.framework`, Likes:
    >> ls Pods/TensorFlowLiteC/Frameworks
    TensorFlowLiteC.framework  TensorFlowLiteCCoreML.framework  TensorFlowLiteCMetal.framework.

## Android Notice

* dynamic linked, auto download aar from [jcenter](https://bintray.com/google/tensorflow/tensorflow-lite). This is
  useful with [cargo-apk](https://github.com/rust-windowing/android-ndk-rs/tree/master/cargo-apk).

## macOS

* env `TENSORFLOWLITE_C_PATH` required for build time.
* env `DYLD_FALLBACK_LIBRARY_PATH` required for runtime.
