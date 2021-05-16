use std::ffi::{c_void, CStr, CString};
use std::mem::size_of;
use std::os::raw::c_char;

use vsprintf::vsprintf;

use tensorflow_lite_sys as ffi;

#[test]
fn c_api_version_test() {
    let version = unsafe {
        let version: *const c_char = ffi::TfLiteVersion();
        CStr::from_ptr(version)
    };
    assert_ne!("", version.to_str().unwrap())
}

// TEST(CApiSimple, Smoke) {
#[test]
fn c_api_smoke_test() {
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

        // std::array<int, 1> input_dims = {2};
        let input_dims = [2i32];
        assert_eq!(
            ffi::TfLiteInterpreterResizeInputTensor(
                interpreter,
                0,
                input_dims.as_ptr() as *const _,
                input_dims.len() as i32
            ),
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
        let tensor_name = ffi::TfLiteTensorName(input_tensor);
        assert_eq!(CStr::from_ptr(tensor_name).to_str().unwrap(), "input");

        let input_params: ffi::TfLiteQuantizationParams =
            ffi::TfLiteTensorQuantizationParams(input_tensor);
        assert_eq!(input_params.scale, 0f32);
        assert_eq!(input_params.zero_point, 0);

        // std::array<float, 2> input = {1.f, 3.f};
        let input = [1f32, 3f32];
        assert_eq!(
            ffi::TfLiteTensorCopyFromBuffer(
                input_tensor,
                input.as_ptr() as *const _ as *const c_void,
                input.len() * size_of::<f32>()
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
        let tensor_name = ffi::TfLiteTensorName(output_tensor);
        assert_eq!(CStr::from_ptr(tensor_name).to_str().unwrap(), "output");

        let output_params: ffi::TfLiteQuantizationParams =
            ffi::TfLiteTensorQuantizationParams(output_tensor);
        assert_eq!(output_params.scale, 0f32);
        assert_eq!(output_params.zero_point, 0);

        // std::array<float, 2> output;
        let mut output = [0f32, 0f32];
        assert_eq!(
            ffi::TfLiteTensorCopyToBuffer(
                output_tensor,
                output.as_mut_ptr() as *mut _ as *mut c_void,
                output.len() * size_of::<f32>()
            ),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(&output, &[3f32, 9f32]);

        ffi::TfLiteInterpreterDelete(interpreter);
    }
}

// TEST(CApiSimple, QuantizationParams) {
#[test]
fn c_api_simple_quantization_params_test() {
    unsafe {
        let path = CString::new("testdata/add_quantized.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        assert!(!options.is_null());

        let interpreter = ffi::TfLiteInterpreterCreate(model, std::ptr::null());
        assert!(!interpreter.is_null());

        ffi::TfLiteModelDelete(model);

        // const std::array<int, 1> input_dims = {2};
        let input_dims = [2i32];
        assert_eq!(
            ffi::TfLiteInterpreterResizeInputTensor(
                interpreter,
                0,
                input_dims.as_ptr() as *const _,
                input_dims.len() as i32
            ),
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
            ffi::TfLiteType_kTfLiteUInt8
        );
        assert_eq!(ffi::TfLiteTensorNumDims(input_tensor), 1);
        assert_eq!(ffi::TfLiteTensorDim(input_tensor, 0), 2);

        let input_params: ffi::TfLiteQuantizationParams =
            ffi::TfLiteTensorQuantizationParams(input_tensor);
        assert_eq!(input_params.scale, 0.003922f32);
        assert_eq!(input_params.zero_point, 0);

        // const std::array<uint8_t, 2> input = {1, 3};
        let input = [1u8, 3u8];
        assert_eq!(
            ffi::TfLiteTensorCopyFromBuffer(
                input_tensor,
                input.as_ptr() as *const _ as *const c_void,
                input.len() * size_of::<u8>()
            ),
            ffi::TfLiteStatus_kTfLiteOk
        );

        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );

        let output_tensor = ffi::TfLiteInterpreterGetOutputTensor(interpreter, 0);
        assert!(!output_tensor.is_null());

        let output_params = ffi::TfLiteTensorQuantizationParams(output_tensor);
        assert_eq!(output_params.scale, 0.003922f32);
        assert_eq!(output_params.zero_point, 0);

        // std::array<uint8_t, 2> output;
        let mut output = [0u8; 2];
        assert_eq!(
            ffi::TfLiteTensorCopyToBuffer(
                output_tensor,
                output.as_mut_ptr() as *mut _ as *mut c_void,
                output.len() * size_of::<u8>()
            ),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(&output, &[3, 9]);

        let dequantizedOutput0 =
            output_params.scale * (output[0] as i32 - output_params.zero_point) as f32;
        let dequantizedOutput1 =
            output_params.scale * (output[1] as i32 - output_params.zero_point) as f32;
        assert_eq!(dequantizedOutput0, 0.011766f32);
        assert_eq!(dequantizedOutput1, 0.035298f32);

        ffi::TfLiteInterpreterDelete(interpreter);
    }
}

// TEST(CApiSimple, Delegate) {
#[test]
fn c_api_delegate_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());

        // Create and install a delegate instance.
        let mut delegate_prepared = false;
        let mut delegate = ffi::TfLiteDelegateCreate();
        delegate.data_ = &mut delegate_prepared as *mut _ as *mut c_void;
        extern "C" fn delegate_prepare(
            _context: *mut ffi::TfLiteContext,
            delegate: *mut ffi::TfLiteDelegate,
        ) -> ffi::TfLiteStatus {
            unsafe { *((*delegate).data_ as *mut bool) = true };
            ffi::TfLiteStatus_kTfLiteOk
        }
        delegate.Prepare = Some(delegate_prepare);
        let options = ffi::TfLiteInterpreterOptionsCreate();
        ffi::TfLiteInterpreterOptionsAddDelegate(options, &mut delegate as *mut _);
        let interpreter = ffi::TfLiteInterpreterCreate(model, options);

        // The delegate should have been applied.
        assert!(delegate_prepared);

        // Subsequent execution should behave properly (the delegate is a no-op).
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk,
        );
        ffi::TfLiteInterpreterDelete(interpreter);
    }
}

// TEST(CApiSimple, DelegateFails) {
#[test]
fn c_api_delegate_fails_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());

        // Create and install a delegate instance.
        let mut delegate = ffi::TfLiteDelegateCreate();
        extern "C" fn delegate_prepare(
            _context: *mut ffi::TfLiteContext,
            _delegate: *mut ffi::TfLiteDelegate,
        ) -> ffi::TfLiteStatus {
            ffi::TfLiteStatus_kTfLiteError
        }
        delegate.Prepare = Some(delegate_prepare);
        let options = ffi::TfLiteInterpreterOptionsCreate();
        ffi::TfLiteInterpreterOptionsAddDelegate(options, &mut delegate as *mut _);
        let interpreter = ffi::TfLiteInterpreterCreate(model, options);

        // Interpreter creation should fail as delegate preparation failed.
        assert!(interpreter.is_null());

        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
    }
}

// TEST(CApiSimple, ErrorReporter) {
#[test]
fn c_api_simple_error_reporter_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        let options = ffi::TfLiteInterpreterOptionsCreate();

        // Install a custom error reporter into the interpreter by way of options.
        #[repr(C)]
        struct TestErrorReporter {
            error_messages: Vec<String>,
        }
        impl TestErrorReporter {
            fn report(&mut self, message: String) {
                self.error_messages.push(message);
            }
            fn num_calls(&self) -> usize {
                self.error_messages.len()
            }
        }
        extern "C" fn report(
            user_data: *mut c_void,
            format: *const c_char,
            args: *mut ffi::va_list,
        ) {
            unsafe { (user_data as *mut TestErrorReporter).as_mut() }
                .unwrap()
                .report(unsafe { vsprintf(format, args).unwrap() });
        }
        let mut reporter = TestErrorReporter {
            error_messages: Vec::new(),
        };
        ffi::TfLiteInterpreterOptionsSetErrorReporter(
            options,
            Some(report),
            &mut reporter as *mut _ as *mut c_void,
        );
        let interpreter = ffi::TfLiteInterpreterCreate(model, options);

        // The options/model can be deleted immediately after interpreter creation.
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);

        // Invoke the interpreter before tensor allocation.
        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteError
        );

        println!("{:?}", &reporter.error_messages);
        // The error should propagate to the custom error reporter.
        assert!(reporter.error_messages[0].starts_with("Invoke called on model that is not ready."));
        assert_eq!(reporter.num_calls(), 1);

        ffi::TfLiteInterpreterDelete(interpreter);
    }
}

// TEST(CApiSimple, ValidModel) {
#[test]
fn c_api_simple_valid_model_test() {
    let bytes = include_bytes!("../testdata/add.bin");
    unsafe {
        let model = ffi::TfLiteModelCreate(bytes.as_ptr() as *const c_void, bytes.len());
        assert!(!model.is_null());
        ffi::TfLiteModelDelete(model);
    }
}

// TEST(CApiSimple, ValidModelFromFile) {
#[test]
fn c_api_simple_valid_model_from_file_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());
        ffi::TfLiteModelDelete(model);
    }
}

// TEST(CApiSimple, InvalidModel) {
#[test]
fn c_api_simple_invalid_model_test() {
    // std::vector<char> invalid_model(20, 'c');
    unsafe {
        let invalid_model = CString::new("cccccccccccccccccccccc").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreate(
            invalid_model.as_ptr() as *const _,
            invalid_model.as_bytes().len(),
        );
        assert!(model.is_null());
    }
}

// TEST(CApiSimple, InvalidModelFromFile) {
#[test]
fn c_api_simple_invalid_model_from_file_test() {
    unsafe {
        let path = CString::new("invalid/path/foo.tflite").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(model.is_null());
    }
}
