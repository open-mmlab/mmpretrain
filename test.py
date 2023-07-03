from mmpretrain import inference_model

result = inference_model('instructblip-vicuna7b_3rdparty-zeroshot_caption',
                         'demo/cat-dog.png')
print(result)
