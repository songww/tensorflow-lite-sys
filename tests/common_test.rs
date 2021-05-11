// use std::ffi::CStr;

use tensorflow_lite_sys as ffi;

/// NOTE: this tests only the TfLiteIntArray part of context.
/// most of common.h is provided in the context of using it with
/// interpreter.h and interpreter.cc, so interpreter_test.cc tests context
/// structures more thoroughly.

// TEST(IntArray, TestIntArrayCreate) {
#[test]
fn test_int_array_creation() {
    unsafe {
        let a: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(0);
        let b: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(3);
        ffi::TfLiteIntArrayFree(a);
        ffi::TfLiteIntArrayFree(b);
    }
}

/*
// TEST(IntArray, TestIntArrayCopy) {
#[test]
fn test_int_array_copy() {
    let a: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(2);
    a.data[0] = 22;
    a.data[1] = 24;
    let b: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCopy(a);
    assert_ne!(*a, *b);
    assert_eq!((*a).size, (*b).size);
    assert_eq!((*a).data[0], (*b).data[0]);
    assert_eq!((*a).data[1], (*b).data[1]);
    ffi::TfLiteIntArrayFree(a);
    ffi::TfLiteIntArrayFree(b);
}

// TEST(IntArray, TestIntArrayEqual) {
#[test]
fn test_int_array_equal() {
    let a: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(1);
    (*a).data[0] = 1;
    let b: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(2);
    (*b).data[0] = 5;
    (*b).data[1] = 6;
    let c: *mut ffi::TfLiteIntArray = ffi::TfLiteIntArrayCreate(2);
    (*c).data[0] = 5;
    (*c).data[1] = 6;
    let d: *mut ffi::TfLiteIntArray = TfLiteIntArrayCreate(2);
    (*d).data[0] = 6;
    (*d).data[1] = 6;
    assert!(!TfLiteIntArrayEqual(a, b));
    assert!(TfLiteIntArrayEqual(b, c));
    assert!(TfLiteIntArrayEqual(b, b));
    assert!(!TfLiteIntArrayEqual(c, d));
    ffi::TfLiteIntArrayFree(a);
    ffi::TfLiteIntArrayFree(b);
    ffi::TfLiteIntArrayFree(c);
    ffi::TfLiteIntArrayFree(d);
}
*/

// TEST(FloatArray, TestFloatArrayCreate) {
#[test]
fn test_float_array_creation() {
    unsafe {
        let a: *mut ffi::TfLiteFloatArray = ffi::TfLiteFloatArrayCreate(0);
        let b: *mut ffi::TfLiteFloatArray = ffi::TfLiteFloatArrayCreate(3);
        ffi::TfLiteFloatArrayFree(a);
        ffi::TfLiteFloatArrayFree(b);
    }
}

/*
// TEST(Types, TestTypeNames) {
#[test]
fn test_type_names() {
    let type_name = |t: ffi::TfLiteType| {
        CStr::from_ptr(ffi::TfLiteTypeGetName(t))
            .to_string_lossy()
            .to_string()
    };
    assert_eq!(&type_name(ffi::kTfLiteNoType), "NOTYPE");
    assert_eq!(&type_name(ffi::kTfLiteFloat64), "FLOAT64");
    assert_eq!(&type_name(ffi::kTfLiteFloat32), "FLOAT32");
    assert_eq!(&type_name(ffi::kTfLiteFloat16), "FLOAT16");
    assert_eq!(&type_name(ffi::kTfLiteInt16), "INT16");
    assert_eq!(&type_name(ffi::kTfLiteInt32), "INT32");
    assert_eq!(&type_name(ffi::kTfLiteUInt32), "UINT32");
    assert_eq!(&type_name(ffi::kTfLiteUInt8), "UINT8");
    assert_eq!(&type_name(ffi::kTfLiteUInt64), "UINT64");
    assert_eq!(&type_name(ffi::kTfLiteInt8), "INT8");
    assert_eq!(&type_name(ffi::kTfLiteInt64), "INT64");
    assert_eq!(&type_name(ffi::kTfLiteBool), "BOOL");
    assert_eq!(&type_name(ffi::kTfLiteComplex64), "COMPLEX64");
    assert_eq!(&type_name(ffi::kTfLiteComplex128), "COMPLEX128");
    assert_eq!(&type_name(ffi::kTfLiteString), "STRING");
    assert_eq!(&type_name(ffi::kTfLiteResource), "RESOURCE");
    assert_eq!(&type_name(ffi::kTfLiteVariant), "VARIANT");
}

TEST(Quantization, TestQuantizationFree) {
  TfLiteTensor t;
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;
  t.quantization.type = kTfLiteAffineQuantization;
  t.sparsity = nullptr;
  auto* params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  params->scale = TfLiteFloatArrayCreate(3);
  params->zero_point = TfLiteIntArrayCreate(3);
  t.quantization.params = reinterpret_cast<void*>(params);
  TfLiteTensorFree(&t);
}

TEST(Sparsity, TestSparsityFree) {
  TfLiteTensor t = {};
  // Set these values, otherwise TfLiteTensorFree has uninitialized values.
  t.allocation_type = kTfLiteArenaRw;
  t.dims = nullptr;
  t.dims_signature = nullptr;

  // A dummy CSR sparse matrix.
  t.sparsity = static_cast<TfLiteSparsity*>(malloc(sizeof(TfLiteSparsity)));
  t.sparsity->traversal_order = TfLiteIntArrayCreate(2);
  t.sparsity->block_map = nullptr;

  t.sparsity->dim_metadata = static_cast<TfLiteDimensionMetadata*>(
      malloc(sizeof(TfLiteDimensionMetadata) * 2));
  t.sparsity->dim_metadata_size = 2;

  t.sparsity->dim_metadata[0].format = kTfLiteDimDense;
  t.sparsity->dim_metadata[0].dense_size = 4;

  t.sparsity->dim_metadata[1].format = kTfLiteDimSparseCSR;
  t.sparsity->dim_metadata[1].array_segments = TfLiteIntArrayCreate(2);
  t.sparsity->dim_metadata[1].array_indices = TfLiteIntArrayCreate(3);

  TfLiteTensorFree(&t);
}
*/
