use std::env;
use std::path::PathBuf;

use bindgen;

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings = bindgen::builder()
        .header("v2.4/tensorflow/lite/c/c_api_experimental.h")
        .header("v2.4/tensorflow/lite/c/common.h")
        .header("v2.4/tensorflow/lite/c/c_api.h")
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .whitelist_var("TfLite.*")
        .whitelist_type("TfLite.*")
        .whitelist_function("TfLite.*")
        .blacklist_type("std.*")
        .bitfield_enum("TfLiteDelegateFlags")
        .enable_cxx_namespaces()
        .enable_function_attribute_detection()
        .clang_arg("-xc++")
        .clang_arg("-std=c++14")
        .clang_arg("-Iv2.4")
        .generate()
        .unwrap();

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
