from mmpretrain import inference_model

result = inference_model('instructblip-vicuna7b_3rdparty-zeroshot_caption', 'demo/cat-dog.png')
print(result)
# {'pred_caption': 'This image shows a small dog and a kitten sitting on a blanket in a field of flowers. The dog is looking up at the kitten with a playful expression on its face. The background is a colorful striped blanket, and there are flowers all around them. The image is well composed with the two animals sitting in the center of the frame, surrounded by the flowers and blanket.'}