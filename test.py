import torch
from argparse import ArgumentParser
from os.path import join, isfile
from os import makedirs
from dataset import TEDDataModule
from model import build_audiofeat_net
from model import FusionNet
from model import AVNet
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np


img_rows, img_cols = 224, 224
sampling_rate = 16000
window_size = 512
window_shift = 128

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    clean_root = join(args.save_root, "clean")
    noisy_root = join(args.save_root, "noisy")
    enhanced_root = join(args.save_root, args.model_uid)
    makedirs(args.save_root, exist_ok=True)
    makedirs(clean_root, exist_ok=True)
    makedirs(noisy_root, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)

    datamodule = TEDDataModule(batch_size=args.batch_size, mask=args.mask, a_only=True)

    if args.dev_set and args.test_set:
        raise RuntimeError("Select either dev set or test set")
    elif args.dev_set:
        dataset = datamodule.dev_dataset
    elif args.test_set:
        dataset = datamodule.test_dataset
    else:
        raise RuntimeError("Select one of dev set and test set")

    if not args.oracle:
        audiofeat_net = build_audiofeat_net(a_only=True)

        fusion_net = FusionNet(a_only=True, mask=args.mask)
        print("Loading model components", args.ckpt_path)

        if args.ckpt_path.endswith("ckpt") and isfile(args.ckpt_path):
            model = AVNet.load_from_checkpoint(args.ckpt_path, nets=(None, audiofeat_net, fusion_net),
                                               loss=args.loss, args=args, a_only=True)
            print("Model loaded")
        else:
            raise FileNotFoundError("Cannot load model weights: {}".format(args.ckpt_path))

        if not args.cpu:
            model.to("cuda:0")
        model.eval()

    i = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            filename = f"{str(i).zfill(5)}.wav"
            clean_path = join(clean_root, filename)
            noisy_path = join(noisy_root, filename)
            enhanced_path = join(enhanced_root, filename)

            if not isfile(clean_path) and not args.test_set:
                sf.write(clean_path, data["clean"], samplerate=sampling_rate)
            if not isfile(noisy_path):
                noisy = librosa.istft(data["noisy_stft"].T, win_length=window_size, hop_length=window_shift,
                                      window="hann", length=len(data["clean"]))
                sf.write(noisy_path, noisy, samplerate=sampling_rate)

            if not isfile(enhanced_path):
                if args.oracle:
                    pred_mag = np.abs(data["noisy_stft"]) * data["mask"].T
                    i += 1
                else:
                    inputs = {"noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(
                        model.device)}

                    pred = model(inputs).cpu()
                    pred_mag = pred.numpy()[0][0]

                noisy_phase = np.angle(data["noisy_stft"])
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated.T, win_length=window_size, hop_length=window_shift,
                                                window="hann", length=len(data["clean"]))
                sf.write(enhanced_path, estimated_audio, samplerate=sampling_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--oracle", type=str2bool, required=False)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model_uid", type=str, required=True)
    parser.add_argument("--dev_set", type=str2bool, required=True)
    parser.add_argument("--test_set", type=str2bool, required=False)
    parser.add_argument("--cpu", type=str2bool, required=False, help="Evaluate model on CPU")
    parser.add_argument("--mask", type=str, default="mag")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="l1")
    args = parser.parse_args()
    main(args)
