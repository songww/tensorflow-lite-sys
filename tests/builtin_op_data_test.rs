use tensorflow_lite_sys as ffi;

// Builtin op data is just a set of data definitions, so the only meaningful
// test we can run is whether we can create the structs we expect to find.
// Testing each struct's members might be possible, but it seems unnecessary
// until we've locked down the API. The build rule has copts set to ignore the
// unused variable warning, since this is just a compilation test.
// TEST(IntArray, CanCompileStructs) {
#[test]
fn test_int_array_can_compile_structs() {
    let padding: ffi::TfLitePadding = ffi::TfLitePadding_kTfLitePaddingSame;
    let padding_values: ffi::TfLitePaddingValues;
    let fused_activation: ffi::TfLiteFusedActivation = ffi::TfLiteFusedActivation_kTfLiteActRelu;
    let conv_params: ffi::TfLiteConvParams;
    let pool_params: ffi::TfLitePoolParams;
    let depthwise_conv_params: ffi::TfLiteDepthwiseConvParams;
    let svdf_params: ffi::TfLiteSVDFParams;
    let rnn_params: ffi::TfLiteRNNParams;
    let sequence_rnn_params: ffi::TfLiteSequenceRNNParams;
    let fully_connected_weights_format: ffi::TfLiteFullyConnectedWeightsFormat =
        ffi::TfLiteFullyConnectedWeightsFormat_kTfLiteFullyConnectedWeightsFormatDefault;
    let fully_connected_params: ffi::TfLiteFullyConnectedParams;
    let projection_type: ffi::TfLiteLSHProjectionType =
        ffi::TfLiteLSHProjectionType_kTfLiteLshProjectionDense;
    let projection_params: ffi::TfLiteLSHProjectionParams;
    let softmax_params: ffi::TfLiteSoftmaxParams;
    let concatenation_params: ffi::TfLiteConcatenationParams;
    let add_params: ffi::TfLiteAddParams;
    let space_to_batch_nd_params: ffi::TfLiteSpaceToBatchNDParams;
    let batch_to_space_nd_params: ffi::TfLiteBatchToSpaceNDParams;
    let mul_params: ffi::TfLiteMulParams;
    let sub_params: ffi::TfLiteSubParams;
    let div_params: ffi::TfLiteDivParams;
    let l2_norm_params: ffi::TfLiteL2NormParams;
    let local_response_norm_params: ffi::TfLiteLocalResponseNormParams;
    let lstm_kernel_type: ffi::TfLiteLSTMKernelType =
        ffi::TfLiteLSTMKernelType_kTfLiteLSTMBasicKernel;
    let lstm_params: ffi::TfLiteLSTMParams;
    let resize_bilinear_params: ffi::TfLiteResizeBilinearParams;
    let pad_params: ffi::TfLitePadParams;
    let pad_v2_params: ffi::TfLitePadV2Params;
    let reshape_params: ffi::TfLiteReshapeParams;
    let skip_gram_params: ffi::TfLiteSkipGramParams;
    let space_to_depth_params: ffi::TfLiteSpaceToDepthParams;
    let depth_to_space_params: ffi::TfLiteDepthToSpaceParams;
    let cast_params: ffi::TfLiteCastParams;
    let combiner_type: ffi::TfLiteCombinerType = ffi::TfLiteCombinerType_kTfLiteCombinerTypeSqrtn;
    let lookup_sparse_params: ffi::TfLiteEmbeddingLookupSparseParams;
    let gather_params: ffi::TfLiteGatherParams;
    let transpose_params: ffi::TfLiteTransposeParams;
    let reducer_params: ffi::TfLiteReducerParams;
    let split_params: ffi::TfLiteSplitParams;
    let split_v_params: ffi::TfLiteSplitVParams;
    let squeeze_params: ffi::TfLiteSqueezeParams;
    let strided_slice_params: ffi::TfLiteStridedSliceParams;
    let arg_max_params: ffi::TfLiteArgMaxParams;
    let arg_min_params: ffi::TfLiteArgMinParams;
    let transpose_conv_params: ffi::TfLiteTransposeConvParams;
    let sparse_to_dense_params: ffi::TfLiteSparseToDenseParams;
    let shape_params: ffi::TfLiteShapeParams;
    let rank_params: ffi::TfLiteRankParams;
    let fake_quant_params: ffi::TfLiteFakeQuantParams;
    let pack_params: ffi::TfLitePackParams;
    let unpack_params: ffi::TfLiteUnpackParams;
    let one_hot_params: ffi::TfLiteOneHotParams;
    let bidi_sequence_rnn_params: ffi::TfLiteBidirectionalSequenceRNNParams;
    let bidi_sequence_lstm_params: ffi::TfLiteBidirectionalSequenceLSTMParams;
}
