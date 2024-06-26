import sys
import torch
import os
import numpy as np
import argparse
import yaml
import logging
import openvino as ov
import struct
import time

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

try:
    from kantts.models import model_builder
    from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
except ImportError:
    raise ImportError("Please install kantts.")

logging.basicConfig(
    #  filename=os.path.join(stage_dir, 'stdout.log'),
    format="%(asctime)s, %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def denorm_f0(mel, f0_threshold=30, uv_threshold=0.6, norm_type='mean_std', f0_feature=None):
    if norm_type == 'mean_std':
        f0_mvn = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * f0_mvn[1:, :] + f0_mvn[0:1, :]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv
    else: # global
        f0_global_max_min = f0_feature

        f0 = mel[:, -2]
        uv = mel[:, -1]

        uv[uv < uv_threshold] = 0.0
        uv[uv >= uv_threshold] = 1.0

        f0 = f0 * (f0_global_max_min[0] - f0_global_max_min[1]) + f0_global_max_min[1]
        f0[f0 < f0_threshold] = f0_threshold

        mel[:, -2] = f0
        mel[:, -1] = uv

    return mel

def am_synthesis(is_ov_convert, symbol_seq, fsnet, ov_model, ling_unit, device, se=None):
    inputs_feat_lst = ling_unit.encode_symbol_sequence(symbol_seq)
    
    inputs_feat_index = 0
    if ling_unit.using_byte():
        inputs_byte_index = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack([inputs_byte_index], dim=-1).unsqueeze(0)
    else:
        inputs_sy = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_tone = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_syllable = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_feat_index = inputs_feat_index + 1
        inputs_ws = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index]).long().to(device)
        )
        inputs_ling = torch.stack(
            [inputs_sy, inputs_tone, inputs_syllable, inputs_ws], dim=-1
        ).unsqueeze(0)

    inputs_feat_index = inputs_feat_index + 1
    inputs_emo = (
        torch.from_numpy(inputs_feat_lst[inputs_feat_index])
        .long()
        .to(device)
        .unsqueeze(0)
    )

    inputs_feat_index = inputs_feat_index + 1
    se_enable = False if se is None else True
    if se_enable:
        inputs_spk = (
            torch.from_numpy(se.repeat(len(inputs_feat_lst[inputs_feat_index]), axis=0))
            .float()
            .to(device)
            .unsqueeze(0)[:, :-1, :]
        )
    else:
        inputs_spk = (
            torch.from_numpy(inputs_feat_lst[inputs_feat_index])
            .long()
            .to(device)
            .unsqueeze(0)[:, :-1]
        )

    inputs_len = (
        torch.zeros(1).to(device).long() + inputs_emo.size(1) - 1
    )  # minus 1 for "~"
    
    input_dict = {
        "inputs_ling": inputs_ling[:, :-1, :],
        "inputs_emotion": inputs_emo[:, :-1],
        "inputs_speaker": inputs_spk,
        "input_lengths": inputs_len
    }
    if is_ov_convert:
        print(inputs_ling[:, :-1, :].shape, inputs_emo[:, :-1].shape, inputs_spk.shape, inputs_len)
        ov_model = ov.convert_model(fsnet, example_input=input_dict)
        ov.save_model(ov_model, 'sambert_encoder.xml')
        return None
    else:
        start = time.time() * 1000
        mid_res = ov_model(input_dict)
        print("-------------Sambert encoder ov model infer for short sentence costs time: %f ms-------------"%(time.time()* 1000 - start))
        text_hid = torch.Tensor(mid_res[0])
        emo_hid = torch.Tensor(mid_res[1])
        spk_hid = torch.Tensor(mid_res[2])
        inter_masks = torch.tensor(mid_res[3])

        start1 = time.time() * 1000
        res = fsnet(
            text_hid,
            emo_hid,
            spk_hid,
            inter_masks,
        )
        print("-------------Sambert rest part of model in torch infer for short sentence costs time: %f ms-------------"%(time.time()* 1000 - start1))
        # res = fsnet(
        #     inputs_ling[:, :-1, :],
        #     inputs_emo[:, :-1],
        #     inputs_spk,
        #     inputs_len,
        # )

        postnet_outputs = res["postnet_outputs"]
        LR_length_rounded = res["LR_length_rounded"]

        valid_length = int(LR_length_rounded[0].item())
        postnet_outputs = postnet_outputs[0, :valid_length, :].cpu().numpy()

        return postnet_outputs



def am_infer(is_ov_convert, sentence, ckpt, output_dir, se_file=None, config=None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print('select cpu device')
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        am_config_file = os.path.join(
            os.path.dirname(os.path.dirname(ckpt)), "config.yaml"
        )
        with open(am_config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    ling_unit = KanTtsLinguisticUnit(config)
    ling_unit_size = ling_unit.get_unit_size()
    config["Model"]["KanTtsSAMBERT"]["params"].update(ling_unit_size)

    se_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("SE", False) 
    se = np.load(se_file) if se_enable else None

    # nsf
    nsf_enable = config["Model"]["KanTtsSAMBERT"]["params"].get("NSF", False) 
    if nsf_enable:
        nsf_norm_type = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_norm_type", "mean_std")
        if nsf_norm_type == "mean_std":
            f0_mvn_file = os.path.join(
                os.path.dirname(os.path.dirname(ckpt)), "mvn.npy"
            )
            f0_feature = np.load(f0_mvn_file)   
        else: # global
            nsf_f0_global_minimum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_minimum", 30.0) 
            nsf_f0_global_maximum = config["Model"]["KanTtsSAMBERT"]["params"].get("nsf_f0_global_maximum", 730.0) 
            f0_feature = [nsf_f0_global_maximum, nsf_f0_global_minimum]

    model, _, _ = model_builder(is_ov_convert, config, device)
    fsnet = model["KanTtsSAMBERT"]
    
    logging.info("Loading checkpoint: {}".format(ckpt))

    if not torch.cuda.is_available(): 
        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(ckpt)
    

    fsnet.load_state_dict(state_dict["model"], strict=False)
   
    results_dir = os.path.join(output_dir, "feat")
    os.makedirs(results_dir, exist_ok=True)
    fsnet.eval()
    ov_model = None
    if not is_ov_convert:
        core = ov.Core()
        ov_model = core.read_model('sambert_encoder.xml')
        ov_model = core.compile_model(ov_model, "AUTO")

    with open(sentence, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            logging.info("Inference sentence: {}".format(line[0]))
            mel_path = "%s/%s_mel.npy" % (results_dir, line[0])
            # dur_path = "%s/%s_dur.txt" % (results_dir, line[0])
            # f0_path = "%s/%s_f0.txt" % (results_dir, line[0])
            # energy_path = "%s/%s_energy.txt" % (results_dir, line[0])
            
            with torch.no_grad():
                mel_post= am_synthesis(is_ov_convert, line[1], fsnet, ov_model, ling_unit, device, se=se)
            
            if mel_post is None:
                return

            # with torch.no_grad():
            #     mel, mel_post, dur, f0, energy = am_synthesis(
            #         line[1], fsnet, ling_unit, device, se=se
            #     )
            if nsf_enable:
                mel_post = denorm_f0(mel_post, norm_type=nsf_norm_type, f0_feature=f0_feature) 

            np.save(mel_path, mel_post)
            # np.savetxt(dur_path, dur)
            # np.savetxt(f0_path, f0)
            # np.savetxt(energy_path, energy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--se_file", type=str, required=False)

    args = parser.parse_args()

    am_infer(args.sentence, args.ckpt, args.output_dir, args.se_file)
