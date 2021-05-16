#![cfg(feature = "experimental")]
// #include "tensorflow/lite/c/c_api_experimental.h"
//
// #include <string.h>
//
// #include <memory>
// #include <vector>
//
// #include <gmock/gmock.h>
// #include <gtest/gtest.h>
// #include "tensorflow/lite/builtin_ops.h"
// #include "tensorflow/lite/c/c_api.h"
// #include "tensorflow/lite/c/common.h"
// #include "tensorflow/lite/delegates/delegate_test_util.h"
// #include "tensorflow/lite/testing/util.h"

// using testing::HasSubstr;
// using tflite::delegates::test_utils::TestDelegate;
use std::ffi::{c_void, CStr, CString};
use std::mem::size_of;
use std::os::raw::c_char;

use vsprintf::vsprintf;

use tensorflow_lite_sys as ffi;

extern "C" fn registration_invoke(
    _: *mut ffi::TfLiteContext,
    _: *mut ffi::TfLiteNode,
) -> ffi::TfLiteStatus {
    ffi::TfLiteStatus_kTfLiteOk
}

fn get_dummy_registration() -> ffi::TfLiteRegistration {
    unsafe {
        let registration: ffi::TfLiteRegistration = ffi::TfLiteRegistration {
            init: None,
            free: None,
            prepare: None,
            invoke: Some(registration_invoke),
            profiling_string: None,
            builtin_code: 0,
            custom_name: std::ptr::null(),
            version: 0,
        };
        registration
    }
}

// TEST(CApiExperimentalTest, Smoke) {
#[test]
fn c_api_experimental_smoke_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        ffi::TfLiteInterpreterOptionsAddBuiltinOp(
            options,
            ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAdd,
            &mut get_dummy_registration() as *mut _,
            1,
            1,
        );
        if cfg!(target_os = "android") {
            ffi::TfLiteInterpreterOptionsSetUseNNAPI(options, true);
        }

        let interpreter: *mut ffi::TfLiteInterpreter = ffi::TfLiteInterpreterCreate(model, options);
        assert!(!interpreter.is_null());
        assert_eq!(
            ffi::TfLiteInterpreterAllocateTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterResetVariableTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );

        ffi::TfLiteInterpreterDelete(interpreter);
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
    }
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
// TEST(CApiExperimentalTest, SelectedBuiltins) {
#[test]
fn c_api_experimental_selected_builtins_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        ffi::TfLiteInterpreterOptionsAddBuiltinOp(
            options,
            ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAdd,
            &mut get_dummy_registration() as *mut _,
            1,
            1,
        );

        let interpreter: *mut ffi::TfLiteInterpreter =
            ffi::TfLiteInterpreterCreateWithSelectedOps(model, options);
        assert!(!interpreter.is_null());
        assert_eq!(
            ffi::TfLiteInterpreterAllocateTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterResetVariableTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );

        ffi::TfLiteInterpreterDelete(interpreter);
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
    }
}

// Test that when using TfLiteInterpreterCreateWithSelectedOps,
// we do NOT get the standard builtin operators by default.
//TEST(CApiExperimentalTest, MissingBuiltin) {
#[test]
fn c_api_experimental_missing_builtin_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        // Install a custom error reporter into the interpreter by way of options.
        //tflite::TestErrorReporter reporter;
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
        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        ffi::TfLiteInterpreterOptionsSetErrorReporter(
            options,
            Some(report),
            &mut reporter as *mut _ as *mut c_void,
        );

        // Create an interpreter with no builtins at all.
        let interpreter: *mut ffi::TfLiteInterpreter =
            ffi::TfLiteInterpreterCreateWithSelectedOps(model, options);

        // Check that interpreter creation failed, because the model contain a buitin
        // op that wasn't supported, and that we got the expected error messages.
        assert!(interpreter.is_null());
        println!("{:?}", &reporter.error_messages);
        assert!(reporter.error_messages[0]
            .starts_with("Didn't find op for builtin opcode 'ADD' version '1'."));
        assert_eq!(reporter.num_calls(), 2);

        ffi::TfLiteInterpreterDelete(interpreter);
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
    }
}

struct OpResolverData {
    called_for_add: bool, //  = false;
}

impl Default for OpResolverData {
    fn default() -> OpResolverData {
        OpResolverData {
            called_for_add: false,
        }
    }
}

// const TfLiteRegistration* MyFindBuiltinOp(void* user_data,
//                                           TfLiteBuiltinOperator op,
//                                           int version) {
unsafe extern "C" fn my_find_builtin_op(
    user_data: *mut c_void,
    op: ffi::TfLiteBuiltinOperator,
    version: i32,
) -> *const ffi::TfLiteRegistration {
    let my_data = unsafe { (user_data as *mut OpResolverData).as_mut().unwrap() };
    if op == ffi::TfLiteBuiltinOperator_kTfLiteBuiltinAdd && version == 1 {
        my_data.called_for_add = true;
        Box::leak(Box::new(get_dummy_registration()))
    } else {
        std::ptr::null()
    }
}

// const TfLiteRegistration* MyFindCustomOp(void*, const char* custom_op,
//                                          int version) {
unsafe extern "C" fn my_find_custom_op(
    _: *mut c_void,
    custom_op: *const c_char,
    version: i32,
) -> *const ffi::TfLiteRegistration {
    if unsafe { CStr::from_ptr(custom_op) }.to_str().unwrap() == "foo" && version == 1 {
        Box::leak(Box::new(get_dummy_registration()))
    } else {
        std::ptr::null()
    }
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
// TEST(CApiExperimentalTest, SetOpResolver) {
#[test]
fn c_api_experimental_set_op_resolver_test() {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();

        let mut my_data: OpResolverData = OpResolverData::default();
        ffi::TfLiteInterpreterOptionsSetOpResolver(
            options,
            Some(my_find_builtin_op),
            Some(my_find_custom_op),
            &mut my_data as *mut _ as *mut c_void,
        );
        assert_eq!(my_data.called_for_add, false);

        let interpreter: *mut ffi::TfLiteInterpreter =
            ffi::TfLiteInterpreterCreateWithSelectedOps(model, options);
        assert!(!interpreter.is_null());
        assert_eq!(
            ffi::TfLiteInterpreterAllocateTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterResetVariableTensors(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(
            ffi::TfLiteInterpreterInvoke(interpreter),
            ffi::TfLiteStatus_kTfLiteOk
        );
        assert_eq!(my_data.called_for_add, true);

        ffi::TfLiteInterpreterDelete(interpreter);
        ffi::TfLiteInterpreterOptionsDelete(options);
        ffi::TfLiteModelDelete(model);
    }
}

// void AllocateAndSetInputs(TfLiteInterpreter* interpreter) {
fn allocate_and_set_inputs(interpreter: *mut ffi::TfLiteInterpreter) {
    // std::array<int, 1> input_dims = {2};
    unsafe {
        let input_dims = [2];
        assert_eq!(
            ffi::TfLiteInterpreterResizeInputTensor(
                interpreter,
                0,
                input_dims.as_ptr(),
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
    }
}

// void VerifyOutputs(TfLiteInterpreter* interpreter) {
fn verify_outputs(interpreter: *mut ffi::TfLiteInterpreter) {
    unsafe {
        let output_tensor: *const ffi::TfLiteTensor =
            ffi::TfLiteInterpreterGetOutputTensor(interpreter, 0);
        assert!(!output_tensor.is_null());
        // std::array<float, 2> output;
        let mut output = [0f32, 0f32];
        assert_eq!(
            ffi::TfLiteTensorCopyToBuffer(
                output_tensor,
                output.as_mut_ptr() as *mut _ as *mut c_void,
                output.len() * size_of::<f32>(),
            ),
            ffi::TfLiteStatus_kTfLiteOk,
        );
        assert_eq!(&output, &[3f32, 9f32]);
    }
}

// void CheckExecution(TfLiteInterpreterOptions* options,
//                     TfLiteStatus expected_first_result,
//                     TfLiteStatus expected_subsequent_results) {
fn check_execution(
    options: *mut ffi::TfLiteInterpreterOptions,
    expected_first_result: ffi::TfLiteStatus,
    expected_subsequent_results: ffi::TfLiteStatus,
) {
    unsafe {
        let path = CString::new("testdata/add.bin").unwrap();
        let model: *mut ffi::TfLiteModel = ffi::TfLiteModelCreateFromFile(path.as_ptr());
        assert!(!model.is_null());

        let interpreter: *mut ffi::TfLiteInterpreter = ffi::TfLiteInterpreterCreate(model, options);
        assert!(!interpreter.is_null());

        allocate_and_set_inputs(interpreter);
        for i in 0..4 {
            let result = ffi::TfLiteInterpreterInvoke(interpreter);
            let expected_result = if i == 0 {
                expected_first_result
            } else {
                expected_subsequent_results
            };
            assert_eq!(result, expected_result);
            if result != ffi::TfLiteStatus_kTfLiteError {
                verify_outputs(interpreter);
            }
        }

        ffi::TfLiteInterpreterDelete(interpreter);
        ffi::TfLiteModelDelete(model);
    }
}

// TEST_F(TestDelegate, NoDelegate) {
#[test]
fn c_api_experimental_nodelegate_test() {
    unsafe {
        let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
        // Execution without any delegate should succeed.
        check_execution(
            options,
            ffi::TfLiteStatus_kTfLiteOk,
            ffi::TfLiteStatus_kTfLiteOk,
        );
        ffi::TfLiteInterpreterOptionsDelete(options);
    }
}

/*
unsafe extern "C" fn simple_delegate_prepare(context: *mut ffi::TfLiteContext, delegate: *mut TfLiteDelegate) -> TfLiteStatus {
    // auto* simple = static_cast<SimpleDelegate*>(delegate->data_);
    let nodes_to_separate: *mut TfLiteIntArray =
        ffi::TfLiteIntArrayCreate(delegate.nodes_.len());
    // Mark nodes that we want in TfLiteIntArray* structure.
    let mut index: i32 = 0;
    for node_index in &delegate.nodes_ {
      nodes_to_separate.data[idx] = node_index;
      index += 1;
      // make sure node is added
      let node: TfLiteNode;
      let reg: *mut TfLiteRegistration;
      context.GetNodeAndRegistration(context, node_index, &node as , &reg);
      if (simple->custom_op_) {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
        TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
      } else {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
      }
    }
    // Check that all nodes are available
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
      int node_index = execution_plan->data[exec_index];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      if (exec_index == node_index) {
        // Check op details only if it wasn't delegated already.
        if (simple->custom_op_) {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        } else {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
        }
      }
    }


// TEST_F(TestDelegate, DelegateNodeInvokeFailure) {
fn c_api_experimental_delegate_node_invoke_failure_test() {
    let mut data = [0i32, 1i32];
    ffi::TfLiteDelegate {
        data_: data.as_mut_ptr() as *mut _ as *mut c_void,
        Prepare: None,
    }
  // Initialize a delegate that will fail when invoked.
    unsafe {
  let delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  // Create another interpreter with the delegate, without fallback.
  let options: *mut ffi::TfLiteInterpreterOptions = ffi::TfLiteInterpreterOptionsCreate();
  ffi::TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  // Execution with the delegate should fail.
  check_execution(options, ffi::TfLiteStatus_kTfLiteError, ffi::TfLiteStatus_kTfLiteError);
  ffi::TfLiteInterpreterOptionsDelete(options);
    }
}
*/

/*
TEST_F(TestDelegate, DelegateNodeInvokeFailureFallback) {
  // Initialize a delegate that will fail when invoked.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  // Create another interpreter with the delegate, with fallback enabled.
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsSetEnableDelegateFallback(options, true);
  CheckExecution(options,
                 // First execution will report DelegateError which indicates
                 // that the delegate failed but fallback succeeded.
                 kTfLiteDelegateError,
                 // Subsequent executions will not use the delegate and
                 // should therefore succeed.
                 ffi::TfLiteStatus_kTfLiteOk);
  TfLiteInterpreterOptionsDelete(options);
}

TEST_F(TestDelegate, TestFallbackWithMultipleDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 0 /**min_ops_per_subset**/,
      true /**fail_node_invoke**/, false /**automatic_shape_propagation**/,
      false /**custom_op**/));
  // Second delegate supports node 1, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate2_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsSetEnableDelegateFallback(options, true);
  CheckExecution(options,
                 // First execution will report DelegateError which indicates
                 // that the delegate failed but fallback succeeded.
                 kTfLiteDelegateError,
                 // Subsequent executions will not use the delegate and
                 // should therefore succeed.
                 ffi::TfLiteStatus_kTfLiteOk);
  TfLiteInterpreterOptionsDelete(options);
}
*/
