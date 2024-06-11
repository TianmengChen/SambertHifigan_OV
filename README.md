# KAN-TTS-OV

In this repo, KAN-TTS is tried to be deployed with OpenVINO.

## Pre-requisite
### Installation of KAN-TTS and OpenVINO
This section is part from the official readme.md of [`KAN-TTS`](https://www.modelscope.cn/models/iic/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary). 

Get the KAN-TTS source code and create conda environment.
```bash
git clone -b develop https://github.com/alibaba-damo-academy/KAN-TTS.git
cd KAN-TTS
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda env create -f environment.yaml
conda activate maas
```

then we install openvino in same environment. If you want specific version of openvino, you can install it by yourself though [`OpenVINO_Doc`](https://docs.openvino.ai/2024/get-started/install-openvino.html?VERSION=v_2024_1_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE).

```bash
pip install openvino
```

### PipeLine of KAN-TTS
Follow the best practice of offical source code with readme in [`KAN-TTS`](https://www.modelscope.cn/models/iic/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary).

After you finish the pipelining of KAN-TTS, you can copy the `res` folder and ckpt files `speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k` you get from pipeline to this project folder.

## Convert torch model to openVINO model
KAN-TTS includes two models: Sambert and Hifigan. Converting a torch model to OpenVINO requires model inputs. So we use `test.txt` as input of Sambert and use the `res` folder as input of Hifigan.

```bash
python kantts/bin/text_to_wav.py --txt test.txt --output_dir res/test_male_ptts_syn --res_zip speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/resource.zip --am_ckpt speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/tmp_am/ckpt/checkpoint_2400200.pth --voc_ckpt speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/orig_model/basemodel_16k/hifigan/ckpt/checkpoint_2400000.pth  --se_file speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/data/se/se.npy --is_ov_convert
```

After a few minutes, you will get two converted openVINO model `sambert_encoder.xml` `sambert_encoder.bin` and `hifigan_t.xml` `hifigan_t.bin`.

<font color=red>**Note**</font>: Sambert can't be fully converted to openvino model at the moment, we only converted the encoder part of Sambert to openvino model in this project.

## Run the inference with openVINO model
Before run the inference, you should delete `res` folder.

```bash
rm -rf res
```

then run the command below.

```bash
python kantts/bin/text_to_wav.py --txt test.txt --output_dir res/test_male_ptts_syn --res_zip speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/resource.zip --am_ckpt speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/tmp_am/ckpt/checkpoint_2400200.pth --voc_ckpt speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/orig_model/basemodel_16k/hifigan/ckpt/checkpoint_2400000.pth  --se_file speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/pretrain_work_dir/data/se/se.npy
```

After a few minutes, you will get the wav file in `res/test_male_ptts_syn`.