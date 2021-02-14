# tensorflow-lite-sys

This crate provides bindings to the raw low-level C API.

## platforms
[✅] With iOS framework(Manual Compilation or CocoaPods).
[] With Android (Manual Compilation or AAR).
[] Manual Compilation(libtensorflow-lite.a)

## iOS Notice
* static linked by default.
* iOS dose not export experimental apis, enable it should with manual compilation framworks.
* Set env `TFLITE_FRAMEWORK_PATH` for path that contains `TensorFlowLiteC.framework`, Likes:
    >> ls Pods/TensorFlowLiteC/Frameworks/
    TensorFlowLiteC.framework  TensorFlowLiteCCoreML.framework  TensorFlowLiteCMetal.framework.

## Android Notice
* dynamic linked [build android](https://www.tensorflow.org/lite/guide/build_android).
