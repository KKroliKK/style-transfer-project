import os

import numpy as np
import torch

from model_reconet.ffmpeg_tools import VideoReader, VideoWriter
from model_reconet.network import ReCoNet
from model_reconet.utils import (
    Dummy,
    nchw_to_nhwc,
    nhwc_to_nchw,
    postprocess_reconet,
    preprocess_for_reconet,
)

input_path = "../../data/model_reconet/input/videoplayback.mp4"
output_path = "../../data/model_reconet/output/output.mp4"
model_path = "../../models/model_reconet/model.pth"
fps = None
batch_size = 2


class ReCoNetModel:
    def __init__(self, state_dict_path, use_gpu=True, gpu_device=None):
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        with self.device():
            self.model = ReCoNet()
            self.model.load_state_dict(
                torch.load(state_dict_path, map_location="cuda:0")
            )
            self.model = self.to_device(self.model)
            self.model.eval()

    def run(self, images):
        assert images.dtype == np.uint8
        assert 3 <= images.ndim <= 4

        orig_ndim = images.ndim
        if images.ndim == 3:
            images = images[None, ...]

        images = torch.from_numpy(images)
        images = nhwc_to_nchw(images)
        images = images.to(torch.float32) / 255

        with self.device():
            with torch.no_grad():
                images = self.to_device(images)
                images = preprocess_for_reconet(images)
                styled_images = self.model(images)
                styled_images = postprocess_reconet(styled_images)
                styled_images = styled_images.cpu()
                styled_images = torch.clamp(styled_images * 255, 0, 255).to(torch.uint8)
                styled_images = nchw_to_nhwc(styled_images)
                styled_images = styled_images.numpy()
                if orig_ndim == 3:
                    styled_images = styled_images[0]
                return styled_images

    def to_device(self, x):
        if self.use_gpu:
            with self.device():
                return x.cuda()
        else:
            return x

    def device(self):
        if self.use_gpu and self.gpu_device is not None:
            return torch.cuda.device(self.gpu_device)
        else:
            return Dummy()


def create_folder_for_file(path):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)


def inference(input_path, output_path, model_path, fps, batch_size):
    model = ReCoNetModel(model_path, use_gpu=True, gpu_device=0)

    reader = VideoReader(input_path, fps=fps)
    create_folder_for_file(output_path)
    writer = VideoWriter(output_path, reader.width, reader.height, reader.fps)

    with writer:
        batch = []

        for frame in reader:
            batch.append(frame)

            if len(batch) == batch_size:
                batch = np.array(batch)
                for styled_frame in model.run(batch):
                    writer.write(styled_frame)

                batch = []

        if len(batch) != 0:
            batch = np.array(batch)
            for styled_frame in model.run(batch):
                writer.write(styled_frame)


if __name__ == "__main__":
    inference()
