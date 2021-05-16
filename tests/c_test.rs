use std::ffi::{c_void, CStr, CString};
use std::mem::size_of;
use std::os::raw::c_char;

use tensorflow_lite_sys as ffi;

#[test]
fn test_version() {
    // Test the TfLiteVersion function.
    let version = unsafe {
        let version: *const c_char = ffi::TfLiteVersion();
        CStr::from_ptr(version)
    };
    println!("Version = {:?}", version);
    if cfg!(feature = "v2.4") {
        assert_eq!(version.to_str().unwrap(), "2.4.0");
    } else if cfg!(feature = "v2.5") {
        assert_eq!(version.to_str().unwrap(), "2.6.0");
    } else {
        assert!(false, "bad version: {:?}", version);
    }
}

#[test]
fn test_smoke_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        assert!(!options.is_null());
        ffi::TfLiteInterpreterOptionsSetNumThreads(options, 2);

        let interpreter: *mut ffi::TfLiteInterpreter = ffi::TfLiteInterpreterCreate(model, options);
        assert!(!interpreter.is_null());

        // The options/model can be deleted immediately after interpreter creation.
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);

        assert_eq!(
            ffi::TfLiteInterpreterAllocateTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(ffi::TfLiteInterpreterGetInputTensorCount(interpreter), 1);
        assert_eq!(ffi::TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

        let input_dims = vec![2];
        assert_eq!(
            ffi::TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims.as_ptr(), 1),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterAllocateTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );

        let input_tensor: *mut ffi::TfLiteTensor =
            ffi::TfLiteInterpreterGetInputTensor(interpreter, 0);
        assert!(!input_tensor.is_null());
        assert_eq!(
            ffi::TfLiteTensorType(input_tensor),
            ffi::TfLiteType_kTfLiteFloat32
        );
        assert_eq!(ffi::TfLiteTensorNumDims(input_tensor), 1);
        assert_eq!(ffi::TfLiteTensorDim(input_tensor, 0), 2);
        assert_eq!(
            ffi::TfLiteTensorByteSize(input_tensor),
            size_of::<f32>() * 2
        );
        assert!(!ffi::TfLiteTensorData(input_tensor).is_null());
        assert_eq!(
            CStr::from_ptr(ffi::TfLiteTensorName(input_tensor)),
            CString::new("input").unwrap().as_c_str()
        );

        let input_params: ffi::TfLiteQuantizationParams =
            ffi::TfLiteTensorQuantizationParams(input_tensor);
        assert_eq!(input_params.scale, 0f32);
        assert_eq!(input_params.zero_point, 0);

        let input = vec![1f32, 3f32];
        assert_eq!(
            ffi::TfLiteTensorCopyFromBuffer(
                input_tensor,
                input.as_ptr() as *const c_void,
                2 * size_of::<f32>()
            ),
            ffi::TfLiteStatus_kTfLiteOk
        );

        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );

        let output_tensor: *const ffi::TfLiteTensor =
            ffi::TfLiteInterpreterGetOutputTensor(interpreter, 0);
        assert!(!output_tensor.is_null());
        assert_eq!(
            ffi::TfLiteTensorType(output_tensor),
            ffi::TfLiteType_kTfLiteFloat32
        );
        assert_eq!(ffi::TfLiteTensorNumDims(output_tensor), 1);
        assert_eq!(ffi::TfLiteTensorDim(output_tensor, 0), 2);
        assert_eq!(
            ffi::TfLiteTensorByteSize(output_tensor),
            size_of::<f32>() * 2
        );
        assert!(!ffi::TfLiteTensorData(output_tensor).is_null());
        assert_eq!(
            CStr::from_ptr(ffi::TfLiteTensorName(output_tensor)),
            CString::new("output").unwrap().as_c_str()
        );

        let output_params: ffi::TfLiteQuantizationParams =
            ffi::TfLiteTensorQuantizationParams(output_tensor);
        assert_eq!(output_params.scale, 0f32);
        assert_eq!(output_params.zero_point, 0);

        let mut output = vec![0f32, 0f32];
        assert_eq!(
            ffi::TfLiteTensorCopyToBuffer(
                output_tensor,
                output.as_mut_ptr() as *mut c_void,
                2 * size_of::<f32>()
            ),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(output[0], 3f32);
        assert_eq!(output[1], 9f32);

        ffi::TfLiteInterpreterDelete(interpreter);
    }
}
