# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

import mmpretrain.models
from mmpretrain.apis import (ImageClassificationInferencer,
                             ImageRetrievalInferencer, get_model)


class MMPreHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        model = get_model(self.config_file, checkpoint, self.device)
        if isinstance(model, mmpretrain.models.ImageClassifier):
            self.inferencer = ImageClassificationInferencer(model)
        elif isinstance(model, mmpretrain.models.ImageToImageRetriever):
            self.inferencer = ImageRetrievalInferencer(model)
        else:
            raise NotImplementedError(
                f'No available inferencer for {type(model)}')
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        results = []
        for image in data:
            results.append(self.inferencer(image)[0])
        return results

    def postprocess(self, data):
        processed_data = []
        for result in data:
            processed_result = {}
            for k, v in result.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    processed_result[k] = v.tolist()
                else:
                    processed_result[k] = v
            processed_data.append(processed_result)
        return processed_data
