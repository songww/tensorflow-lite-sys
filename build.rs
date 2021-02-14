use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use bindgen;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tfversion = if cfg!(feature = "v2.4") {
        "v2.4"
    } else {
        panic!("Unsupported tensorflow version.");
    };
    let mut builder = bindgen::builder()
        .header(format!("{}/tensorflow/lite/c/common.h", tfversion))
        .header(format!("{}/tensorflow/lite/c/c_api.h", tfversion));

    if cfg!(feature = "xnnpack") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h",
            tfversion
        ));
    }

    if cfg!(feature = "experimental") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/c/c_api_experimental.h",
            tfversion
        ));
    }

    if cfg!(feature = "gpu") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/delegates/gpu/delegate.h",
            tfversion
        ));
    }

    if cfg!(feature = "metal") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/delegates/gpu/metal_delegate.h",
            tfversion
        ));
    }

    if cfg!(feature = "coreml") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/delegates/coreml/coreml_delegate.h",
            tfversion
        ));
    }

    let bindings = builder
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .bitfield_enum("TfLiteDelegateFlags")
        .whitelist_var("TfLite.*")
        .whitelist_type("TfLite.*")
        .whitelist_function("TfLite.*")
        .whitelist_var("TFLGpu.*")
        .whitelist_type("TFLGpu.*")
        .whitelist_function("TFLGpu.*")
        .enable_function_attribute_detection()
        .clang_arg("-xc++")
        .clang_arg("-std=c++14")
        .clang_arg(format!("-I{}", tfversion))
        .generate()
        .unwrap();

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();

    if cfg!(target_os = "ios") {
        let coreml_required = cfg!(feature = "coreml");
        let metal_required = cfg!(feature = "metal");

        if let Ok(frameworks) = env::var("TFLITE_FRAMEWORK_PATH") {
            let frameworks: &Path = frameworks.as_ref();
            if !frameworks.join("TensorFlowLiteC.framework").exists() {
                panic!(
                    "`{}` dose not contains TensorFlowLiteC.framework",
                    frameworks.display()
                )
            }
            ios_framework_to_staticlib(
                frameworks,
                coreml_required,
                metal_required,
                &out_path.join("libtensorflow-lite.a"),
            );
        } else {
            if let Some(pods_root) = find_pods_root() {
                let frameworks = pods_root.join("TensorFlowLiteC").join("Frameworks");
                if !frameworks.join("TensorFlowLiteC.framework").exists() {
                    panic!("TensorFlowLiteC.framework dose not found, Please set env `TFLITE_FRAMEWORK_PATH` that contains the frameworks.")
                }
                ios_framework_to_staticlib(
                    &frameworks,
                    coreml_required,
                    metal_required,
                    &out_path.join("libtensorflow-lite.a"),
                );
            } else {
                panic!("TensorFlowLiteC.framework dose not found, Please set env `TFLITE_FRAMEWORK_PATH` that contains the frameworks.")
            }
        }
        println!("cargo:rustc-link-lib=static=tensorflow-lite");
        println!("cargo:rustc-link-search=native={}", out_path.display());
    }
}

fn find_pods_root() -> Option<PathBuf> {
    if let Ok(pods_root) = env::var("PODS_ROOT") {
        let p: &Path = pods_root.as_ref();
        return Some(p.to_path_buf());
    }
    if let Ok(project_root) = env::var("PROJECT_DIR").or_else(|_| env::var("SRCROOT")) {
        let p: &Path = project_root.as_ref();
        let pods_root = p.join("Pods");
        if pods_root.exists() {
            return Some(pods_root);
        }
    }
    if let Ok(project_root) = env::var("CARGO_MANIFEST_DIR") {
        let p: &Path = project_root.as_ref();
        let pods_root = p.join("Pods");
        if pods_root.exists() {
            return Some(pods_root);
        }
        let pods_root = p.join("..").join("Pods");
        if pods_root.exists() {
            return pods_root.canonicalize().ok();
        }
    }
    return None;
}

fn ios_framework_to_staticlib(
    frameworks: &Path,
    coreml_required: bool,
    metal_required: bool,
    output: &Path,
) {
    let c = frameworks
        .join("TensorFlowLiteC.framework")
        .join("TensorFlowLiteC");
    let mut objects = vec![c.as_os_str()];
    let coreml = frameworks
        .join("TensorFlowLiteCCoreML.framework")
        .join("TensorFlowLiteCCoreML");
    if coreml_required {
        if coreml.exists() {
            objects.push(coreml.as_os_str());
        } else {
            panic!("TensorFlowLiteCCoreML.framework dose not exists.");
        }
    }
    let metal = frameworks
        .join("TensorFlowLiteCMetal.framework")
        .join("TensorFlowLiteCMetal");
    if metal_required {
        if metal.exists() {
            objects.push(metal.as_os_str());
        } else {
            panic!("TensorFlowLiteCMetal.framework dose not exists.");
        }
    }
    Command::new("libtool")
        .arg("-static")
        .args(&objects)
        .arg("-o")
        .arg(output.as_os_str())
        .output()
        .expect("Failed to create libtensorflow-lite.a");
}
