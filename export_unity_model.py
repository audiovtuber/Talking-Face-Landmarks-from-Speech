from argparse import ArgumentParser
from pathlib import Path

import torch

from train import TalkingFaceLSTM

class TalkingFaceUnity(TalkingFaceLSTM):
    
    def transform_landmarks_to_texture(outputs:torch.tensor, resolution:int=128):
        """
        Based of off `overlay_face.py`'s `overlay_landmarks` function. Its purpose is to change the output of the model to a texture 
        that can be directly used by Unity, initially for debugging purposes
        """
        print(f"Outputs: {type(outputs)}")
        outputs = (128 * outputs).long()
        texture_tensor = torch.zeros((outputs.shape[0], 128, 128, 3)).long()
        color_tensor = torch.LongTensor([255, 255, 255])
        for batch_idx, sample in enumerate(outputs):  # one sample of the batch
            for part in sample:  # one landmark (of 68 per sample)
                texture_tensor[batch_idx, part[1]-2:part[1]+2, part[0]-2:part[0]+2] = color_tensor
        
        return texture_tensor


    def forward(self, X, hidden=None, cell=None):
        if hidden is not None:
            lstm_features, (final_hidden, final_cell) = self.lstm_layers(X, (hidden, cell))
        else:
            lstm_features, (final_hidden, final_cell) = self.lstm_layers(X)
        
        outputs = self.output_layer(lstm_features).reshape(X.shape[0], -1, 2)  # (batch, 136)
        
        # TODO: don't be hardcoded
        outputs = self.transform_landmarks_to_texture(outputs)
        return outputs, (final_hidden, final_cell)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', required=False, help='Path to a trained audio-vtuber model. If provided, the resulting video will be augmented with its predictions')
    parser.add_argument('--save-path', help='Where to save the resulting video')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = TalkingFaceUnity.load_from_checkpoint(args.model)
    model.to_onnx(Path(args.save_path) / 'trained_model.onnx', torch.randn(1, 128), export_params=True)
    