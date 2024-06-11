import os
import sys
import argparse
import torch
import soundfile as sf
import yaml
import logging
import numpy as np
import time
import glob
import openvino as ov

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.utils.log import logging_to_file
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(ckpt, config=None):
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(os.path.dirname(ckpt))
        config = os.path.join(dirname, "config.yaml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    from kantts.models.hifigan.hifigan import Generator

    model = Generator(**config["Model"]["Generator"]["params"])
    states = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(states["model"]["generator"])

    # add pqmf if needed
    if config["Model"]["Generator"]["params"]["out_channels"] > 1:
        # lazy load for circular error
        from kantts.models.pqmf import PQMF

        model.pqmf = PQMF()

    return model


def binarize(mel, threshold=0.6):
    # vuv binarize
    res_mel = mel.copy()
    index = np.where(mel[:, -1] < threshold)[0]
    res_mel[:, -1] = 1.0
    res_mel[:, -1][index] = 0.0
    return res_mel


def hifigan_infer(is_ov_convert, input_mel, ckpt_path, output_dir, config=None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
        )
        if not os.path.exists(config_path):
            raise ValueError("config file not found: {}".format(config_path))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # check directory existence
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging_to_file(os.path.join(output_dir, "stdout.log"))

    if os.path.isfile(input_mel):
        mel_lst = [input_mel]
    elif os.path.isdir(input_mel):
        mel_lst = glob.glob(os.path.join(input_mel, "*.npy"))
    else:
        raise ValueError("input_mel should be a file or a directory")

    model = load_model(ckpt_path, config)

    logging.info(f"Loaded model parameters from {ckpt_path}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    if not is_ov_convert:
        core = ov.Core()
        ov_model = core.read_model('hifigan_t.xml')
        ov_model = core.compile_model(ov_model, "AUTO")
        infer_request = ov_model.create_infer_request()
        
        
        
    with torch.no_grad():
        start = time.time()
        pcm_len = 0
        for mel in mel_lst:
            start1 = time.time()* 1000
            utt_id = os.path.splitext(os.path.basename(mel))[0]
            mel_data = np.load(mel)
            if model.nsf_enable:
                mel_data = binarize(mel_data)
            # generate
            mel_data = torch.tensor(mel_data, dtype=torch.float).to(device)
            # (T, C) -> (B, C, T)
            mel_data = mel_data.transpose(1, 0).unsqueeze(0)
            if  is_ov_convert:
                ov_model = ov.convert_model(model, example_input=mel_data)
                ov.save_model(ov_model, 'hifigan_t.xml')
                return
            else:
                mel_data = mel_data.numpy()
                ov_output = infer_request.infer(mel_data, share_inputs=True)
                # ov_output = ov_model(mel_data)
                # y = model(mel_data)
                if hasattr(model, "pqmf"):
                    y = model.pqmf.synthesis(y)
                
                y = ov_output[0]
                y = np.squeeze(y)
                print("-------------Hifigan ov model infer for short sentences costs time: %f ms-------------"%(time.time()* 1000 - start1))
                # y = y.view(-1).cpu().numpy()
                pcm_len += len(y)
                

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(output_dir, f"{utt_id}_gen_ov.wav"),
                y,
                config["audio_config"]["sampling_rate"],
                "PCM_16",
            )
            rtf = (time.time() - start) / (
                pcm_len / config["audio_config"]["sampling_rate"]
            )
    if not is_ov_convert:
        
        # report average RTF
        logging.info(
            f"Finished generation of {len(mel_lst)} utterances (RTF = {rtf:.03f})."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer hifigan model")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_mel",
        type=str,
        required=True,
        help="Path to input mel file or directory containing mel files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    hifigan_infer(
        args.input_mel,
        args.ckpt,
        args.output_dir,
        args.config,
    )
