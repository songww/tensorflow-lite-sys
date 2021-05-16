use std::env;
// use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

use zip;

use bindgen;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tfversion = if cfg!(feature = "v2.4") {
        "v2.4"
    } else if cfg!(feature = "v2.5") {
        "v2.5"
    } else {
        panic!("Unsupported tensorflow version.");
    };
    let mut builder = bindgen::builder()
        .header(format!("{}/tensorflow/lite/c/common.h", tfversion))
        .header(format!("{}/tensorflow/lite/c/c_api.h", tfversion));

    if cfg!(feature = "v2.5") {
        builder = builder
            .header(format!("{}/tensorflow/lite/c/builtin_op_data.h", tfversion))
            .header(format!("{}/tensorflow/lite/c/c_api_types.h", tfversion))
            .header(format!("{}/tensorflow/lite/builtin_ops.h", tfversion));
    }
    if cfg!(feature = "external") {
        builder = builder.header(format!(
            "{}/tensorflow/lite/c/external/external_delegate.h",
            tfversion
        ));
    }

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

    if cfg!(feature = "hexagon") {
        if target_os != "android" {
            panic!("'hexagon' delegate only works on 'android' platform!");
        }
        builder = builder.header(format!(
            "{}/tensorflow/lite/delegates/hexagon/hexagon_delegate.h",
            tfversion
        ));
        // TODO: add link of hexagon library.
    }

    let bindings = builder
        .allowlist_var("TfLite.*")
        .allowlist_type("TfLite.*")
        .allowlist_function("TfLite.*")
        .allowlist_var("TFLGpu.*")
        .allowlist_type("TFLGpu.*")
        .allowlist_function("TFLGpu.*")
        .array_pointers_in_arguments(true)
        .bitfield_enum("TfLiteDelegateFlags")
        .bitfield_enum("TfLiteGpuExperimentalFlags")
        .clang_arg("-xc++")
        .clang_arg("-std=c++14")
        .clang_arg(format!("-I{}", tfversion))
        .disable_name_namespacing()
        .enable_function_attribute_detection()
        // .emit_builtins()
        .opaque_type("TfLiteIntArray")
        .opaque_type("TfLiteFloatArray")
        // .prepend_enum_name(false)
        .respect_cxx_access_specs(true)
        .rustfmt_bindings(true)
        .size_t_is_usize(true)
        .generate()
        .unwrap();

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_os == "ios" {
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
    } else if target_os == "android" {
        let lib_search_parh = if let Ok(lib_search_parh) = env::var("TENSORFLOWLITE_C_PATH") {
            PathBuf::from(lib_search_parh)
        } else {
            let version = if cfg!(feature = "v2.4") {
                "2.4.0"
            } else if cfg!(feature = "v2.5") {
                "2.5.0"
            } else {
                unreachable!()
            };
            let out_dir = out_path.join("tensorflow-lite");
            if !out_dir.exists() {
                std::fs::create_dir(&out_dir)
                    .expect(&format!("Can not create dir `{}`", out_dir.display()));
            }
            download_and_extract_aar(version, &out_dir, None);
            if cfg!(feature = "gpu") {
                download_and_extract_aar(version, &out_dir, Some("gpu"));
                // println!("cargo:rustc-link-lib=dylib=tensorflowlite_jni");
            }
            if cfg!(feature = "hexagon") {
                download_and_extract_aar(version, &out_dir, Some("hexagon"));
                // println!("cargo:rustc-link-lib=dylib=tensorflowlite_jni");
            }
            if cfg!(feature = "select-tf-ops") {
                download_and_extract_aar(version, &out_dir, Some("select-tf-ops"));
                // println!("cargo:rustc-link-lib=dylib=tensorflowlite_jni");
            }
            out_dir
        };
        let arch = if target_arch == "x86" {
            "x86"
        } else if target_arch == "x86_64" {
            "x86_64"
        } else if target_arch == "arm" {
            "armeabi-v7a"
        } else if target_arch == "aarch64" {
            "arm64-v8a"
        } else {
            panic!("Unsupported target_arch {}", target_arch);
        };
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_jni");
        println!(
            "cargo:rustc-link-search={}",
            lib_search_parh.join(arch).display()
        );
    } else if target_os == "macos" {
        let lib_search_parh = env::var("TENSORFLOWLITE_C_PATH").expect("Please set env `ENSORFLOWLITE_C_PATH` that contains `libtensorflowlite_c.dylib` for macOS.");
        if !PathBuf::from(&lib_search_parh)
            .join("libtensorflowlite_c.dylib")
            .exists()
        {
            panic!(
                "`libtensorflowlite_c.dylib` dose not found in `{0}`, of env `TENSORFLOWLITE_C_PATH`.",
                lib_search_parh
            );
        }
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
        println!("cargo:rustc-link-search={}", lib_search_parh);
    } else if target_os == "linux" {
        let lib_search_parh = env::var("TENSORFLOWLITE_C_PATH").expect("Please set env `ENSORFLOWLITE_C_PATH` that contains `libtensorflowlite_c.so` for linux.");
        if !PathBuf::from(&lib_search_parh)
            .join("libtensorflowlite_c.so")
            .exists()
        {
            panic!(
                "`libtensorflowlite_c.so` dose not found in `{0}`, of env `TENSORFLOWLITE_C_PATH`.",
                lib_search_parh
            );
        }
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
        println!("cargo:rustc-link-search={}", lib_search_parh);
    }
}

fn download_and_extract_aar(version: &str, out_dir: &PathBuf, feature: Option<&str>) {
    let name = if feature.is_some() {
        format!("tensorflow-lite-{}", feature.unwrap())
    } else {
        "tensorflow-lite".to_string()
    };
    let url = format!(
        "https://repo1.maven.org/maven2/org/tensorflow/{name}/{version}/{name}-{version}.aar",
        name = name,
        version = version
    );
    let aar = out_dir.join(format!("{}-{}.zip", name, version));
    let output = Command::new("curl")
        .arg(&url)
        .arg("-v")
        .arg("-o")
        .arg(&aar)
        .output()
        .expect(&format!(
            "Failed to download {}-{}.aar from `{}`",
            name, version, url
        ));
    if !output.status.success() {
        println!(
            "cargo:warning=stdout {}",
            String::from_utf8_lossy(&output.stdout)
        );
        println!(
            "cargo:warning=stderr {}",
            String::from_utf8_lossy(&output.stderr)
        );
        panic!(
            "Failed to download `{}-{}.aar` from `{}`",
            name, version, url
        );
    }
    println!("cargo:warning=downloaded {}", aar.display());
    let mut zipfile = zip::read::ZipArchive::new(
        std::fs::File::open(&aar).expect(&format!("Can not open `{}`", aar.display())),
    )
    .expect(&format!(
        "Invalid `{}-{}.aar` which was downloaded from `{}`",
        name, version, url
    ));
    zipfile
        .extract(out_dir.join("tensorflow-lite"))
        .expect(&format!("Can not extract `{}-{}.aar`", name, version));
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
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let c = frameworks
        .join("TensorFlowLiteC.framework")
        .join("TensorFlowLiteC");
    let mut objects = match lipo_info(&c) {
        MachOType::Fat(archs) => {
            assert!(archs.contains(&target_arch));
            let thin_object = out_path.join("tensorflow-lite-c.o");
            lipo_thin_arch(&c, &target_arch, &thin_object);
            vec![thin_object]
        }
        MachOType::NonFat(arch) => {
            assert_eq!(arch, target_arch);
            vec![c]
        }
    };
    let coreml = frameworks
        .join("TensorFlowLiteCCoreML.framework")
        .join("TensorFlowLiteCCoreML");
    if coreml_required {
        if !coreml.exists() {
            panic!("TensorFlowLiteCCoreML.framework dose not exists.");
        }
        match lipo_info(&coreml) {
            MachOType::Fat(archs) => {
                assert!(archs.contains(&target_arch));
                let thin_object = out_path.join("tensorflow-lite-coreml.o");
                lipo_thin_arch(&coreml, &target_arch, &thin_object);
                objects.push(thin_object);
            }
            MachOType::NonFat(arch) => {
                assert_eq!(arch, target_arch);
                objects.push(coreml);
            }
        }
    }
    let metal = frameworks
        .join("TensorFlowLiteCMetal.framework")
        .join("TensorFlowLiteCMetal");
    if metal_required {
        if !metal.exists() {
            panic!("TensorFlowLiteCMetal.framework dose not exists.");
        }
        match lipo_info(&metal) {
            MachOType::Fat(archs) => {
                assert!(archs.contains(&target_arch));
                let thin_object = out_path.join("tensorflow-lite-metal.o");
                lipo_thin_arch(&metal, &target_arch, &thin_object);
                objects.push(thin_object);
            }
            MachOType::NonFat(arch) => {
                assert_eq!(arch, target_arch);
                objects.push(metal);
            }
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

enum MachOType {
    Fat(Vec<String>), // architectures contains.
    NonFat(String),
}

fn lipo_info(binary: &Path) -> MachOType {
    let out = Command::new("lipo")
        .arg("-i")
        .arg(binary.as_os_str())
        .output()
        .expect("Failed to get info of Mach-O binary.");
    let stdout = String::from_utf8_lossy(&out.stdout);
    if stdout.trim().starts_with("Non-fat file:") {
        let mut split = stdout.rsplitn(1, ": ");
        let arch = split.next().expect("Invalid MachO binary architecture.");
        MachOType::NonFat(arch.to_string())
    } else if stdout.trim().starts_with("Architectures in the fat file:") {
        let mut split = stdout.rsplitn(1, ": ");
        let archs = split.next().expect("Invalid MachO binary architecture.");
        MachOType::Fat(
            archs
                .split_ascii_whitespace()
                .map(|arch| arch.to_string())
                .collect(),
        )
    } else {
        panic!("Invalid Mach-O binary.")
    }
}

fn lipo_thin_arch(input: &Path, arch: &str, output: &Path) {
    Command::new("lipo")
        .arg("-thin")
        .arg(&arch)
        .arg("-output")
        .arg(output.as_os_str())
        .arg(input.as_os_str())
        .status()
        .expect(&format!(
            "Failed to thin {} from Mach-O binary {}.",
            &arch,
            input.display()
        ));
}
