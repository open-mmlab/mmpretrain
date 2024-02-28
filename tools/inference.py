from pathlib import Path
from argparse import ArgumentParser
import os
import json
import time
from mmpretrain import ImageClassificationInferencer

class FPSLogger:
    def __init__(self, num_of_images):
        self.tottime = 0.0
        self.count = 0
        self.last_record = 0.0
        self.last_print = time.time()
        self.interval = 3
        self.num_of_images = num_of_images

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.tottime += time.time() - self.last_record
        self.count += 1
        self.print_fps()

    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - mmpret - INFO - Predict({self.count}/{self.num_of_images}) "
                  f"- Inference running at {self.count / self.tottime:.3f} FPS")
            self.last_print = time.time()

def main(args):
    fps_logger = FPSLogger(len(os.listdir(args.images_dir)))
    inference = ImageClassificationInferencer(
        model=args.config,
        pretrained=args.checkpoint,
    )
    if args.silent:
        inference.show_progress = False

    images: Path = args.images_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    image_paths = list(images.glob("**/*.jpg"))
    for i, p in enumerate(image_paths, 1):
        try:
            fps_logger.start_record()
            result = inference(str(p))
            fps_logger.end_record()

            file_name = str(p).split("/")[-1].replace(".jpg", ".json")
            pred_path = os.path.join(output_dir, file_name)
            prediction = {
                "result": [
                    {
                        "type": "choices",
                        "value": {"choices": [result[0]['pred_class']]},
                        "origin": "manual",
                        "to_name": "image",
                        "from_name": "choice",
                    }
                ],
            }

            with open(pred_path, "w") as f:
                json.dump(prediction, f)
        except Exception as e:
            print(f"Failed with {p}. {e}")
    print(f"Inference time: {round(time.time() - start_time, 2)} s.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--silent",
        action="store_true",
        help="suppress progress bars and verbose output")
    config = parser.parse_args()
    main(config)
