import MinkowskiEngine as ME


def create_input_batch(batch, device="cuda", quantization_size=None,
                       speed_optimized=True, quantization_mode='random'):
    if quantization_size is not None:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    batch["coordinates"] = batch["coordinates"].int()
    if quantization_mode == 'random':
        quantization_mode = ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
    elif quantization_mode == 'avg':
        quantization_mode = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    else:
        raise NotImplementedError
    in_field = ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        quantization_mode=quantization_mode,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED if speed_optimized else ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
        device=device,
    )
    return in_field.sparse()
