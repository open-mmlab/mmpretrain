import numpy as np
import onnxruntime as ort

from mmcls.models.classifiers import BaseClassifier


class ONNXRuntimeClassifier(BaseClassifier):
    """Wrapper for classifier's inference with ONNXRuntime."""

    def __init__(self, onnx_file, class_names, device_id):
        super(ONNXRuntimeClassifier, self).__init__()
        sess = ort.InferenceSession(onnx_file)

        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})
        sess.set_providers(providers, options)

        self.sess = sess
        self.CLASSES = class_names
        self.device_id = device_id
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        self.is_cuda_available = is_cuda_available

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, imgs, img_metas, **kwargs):
        input_data = imgs
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available:
            input_data = input_data.cpu()
        self.io_binding.bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        results = self.io_binding.copy_outputs_to_cpu()[0]
        return list(results)
