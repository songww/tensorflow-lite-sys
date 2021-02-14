use std::env;
use std::path::PathBuf;

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
}
