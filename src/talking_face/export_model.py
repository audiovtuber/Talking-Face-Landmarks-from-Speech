from argparse import ArgumentParser

from talking_face.train import export_to_torchscript


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a trained audio-vtuber model",
    )
    parser.add_argument("--output-dir", help="Where to save the resulting model(s)")
    parser.add_argument("--input-dims", type=int, nargs="*", default=(1, 128), help="The dimensions of the model inputs as comma-separated integers. Defaults to (1, 128)")
    parser.add_argument(
        "target_frameworks",
        nargs="+",
        help="Which framework(s) to target. Can be 'torchscript'")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.target_frameworks = list(map(str.lower, args.target_frameworks))

    # NOTE: model is not currently ONNX-compatible
    # if 'onnx' in args.target_frameworks:
    #     export_to_onnx(model=args.model, output_dir=args.output_dir, input_dims=args.input_dims)

    if 'torchscript' in args.target_frameworks:
        export_to_torchscript(model=args.model, output_dir=args.output_dir)
