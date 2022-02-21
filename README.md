# 202109_ohtani_imgaug

品質を落とした動画の認識率を測る．

```bash
.
├── .gitignore
├── README.md
├── acc_fig/　# 認識率の画像保存ディレクトリ
├── batch_video_check.ipynb
├── class_check/
│   ├── gop/
│   └── original/
├── class_check.ipynb
├── class_check.py
├── clipsampler.ipynb
├── comet_api_figure/
│   ├── comet_api_1.ipynb             # cometのPython APIを使ったJPEG圧縮の圧縮強度0から100までの画質評価のグラフ作成
│   ├── comet_api_2.ipynb             # cometのPython APIを使ったJPEG圧縮の圧縮強度80から100までの画質評価のグラフ作成
│   ├── comet_api_3.ipynb             # cometのPython APIを使ったJPEG圧縮の圧縮強度0から100までの認識率のグラフ作成
│   ├── comet_api_4.ipynb             # cometのPython APIを使ったJPEG圧縮の圧縮強度80から100までの認識率のグラフ作成
│   ├── comet_api_debug.ipynb         # cometのPython APIの動作確認
│   ├── comet_api_ffmpeg_1.ipynb      # cometのPython APIを使った MPEG圧縮した動画の認識率のグラフ作成
│   ├── jpeg_quality_plotfit.ipynb    # imgaugの圧縮強度とquality（PILのimage.save）の関係
│   └── kinetics_train.ipynb          # MPEG圧縮した動画を使って学習したときのグラフ作成（損失，認識率）
├── compression_dataset.py          
├── compression_train.py              # 学習用コード（事前学習済み）
├── compression_train_scratch.py      # 学習用コード（scratch）
├── compression_transform.py          # 検証用のSameClipSamplerクラス，JPEG圧縮での画質評価用に2つの画像を返すためのクラス 
├── config.ini                        # データセットごとのパスのデフォルトなど
├── dataset_imgaug.py                 # 認識率を測るメインのスクリプト
├── dataset_info/                     # MPEG圧縮した動画のファイルサイズやビットレートに関するディレクトリ
│   ├── filesize_fig/                 # ファイルサイズのヒストグラム画像保存用ディレクトリ
│   ├── kinetics400_crfg_filesize.ipynb # ファイルサイズのヒストグラム作成（CRF,GOP指定）
│   ├── kinetics400_crfg_stream_stats.ipynb # ビットレートのヒストグラム作成（CRF,GOP指定）
│   ├── kinetics400_q_filesize.ipynb  # ファイルサイズのヒストグラム作成（q指定）
│   ├── kinetics400_q_stream_stats.ipynb  # ビットレートのヒストグラム作成（q指定）
│   ├── kinetics400stats.ipynb
│   ├── kinetics400stats.txt
│   └── stream_fig/                   # ビットレートのヒストグラム画像保存用ディレクトリ
├── images/
├── img_score.py                      # 画質評価関数(psnrなど)
├── kinetics_train/                   # Kinetics400で学習する用のモデル構築（モデルのLinearを使用）
│   ├── __pycache__/
│   ├── resnet.py
│   ├── slowfast_r101.py
│   └── x3d_m.py
├── model.ini                         # モデルごとのサイズパラメータ（フレーム数，リサイズなど）
├── model_load_val.py                 # 保存したモデルを呼び出して認識率を測る
├── model_path/                       # 学習したのモデルパスディレクトリ
│   ├── .DS_Store
│   ├── ._.DS_Store
│   ├── Kinetics400/
│   ├── UCF101/
│   └── scratch/
├── pytorchvideo_dir.txt              # ディレクト構造
├── score_cal.py                      # 画質評価関数(psnrなど)
├── ucf_model.py
├── ucf_train/                        # Kinetics400以外のデータセットで学習する用のモデル構築（Linearを作成）
│   ├── __pycache__/
│   ├── compression_transform.py
│   ├── model_save.py
│   ├── resnet.py
│   ├── slowfast_r101.py
│   ├── train_ucf.py
│   └── x3d_m.py
├── ucf_train.ipynb
```

## usage

例：dataset_imgaug.py

```:bash
python dataset_imgaug.py --dataset Kinetics400 
python dataset_imgaug.py --dataset Kinetics400 --use_compression -cs 100 --model_name x3d_m 
python dataset_imgaug.py --dataset Kinetics400 --path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/20/ --crf 20 --gop 12
```

例：compression_train.py

```:bash
python compression_train.py --train_data_path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/20/ --train_crf 20 --train_gop 12 --val_data_path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/20/ --val_crf 20 --val_gop 12 --model_name x3d_m --model_pth gop12crf20
python compression_train.py --train_data_path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/30/ --train_crf 30 --train_gop 12 --val_data_path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/30/ --val_crf 30 --val_gop 12 --model_name slow_r50 --model_pth gop12crf30
```

v.py

```:bash
python model_load_val.py --mod_pth original_best.pth --model_name slowfast_r101
python model_load_val.py --path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/12/crf/30/ --crf 30 --gop 12 --mod_pth gop12crf30_best.pth
```
## オプション
#### dataset_imgaug.py

- `--dataset`：datasetの名前．例：Kinetics400,  UCF101
- `--path`：ロードするデータセットのパスを指定．デフォルト以下のようなる．
  - Kinetics400 : /mnt/dataset/Kinetics400/
  - UCF101 : /mnt/dataset/UCF101/
- `--use_compression`：JPEG圧縮をかけるかどうか．圧縮をかける場合はオプションが必要．
- `--compression_strength`：JPEG圧縮の圧縮強度の指定[0 ~ 100]．0が高画質で100が低画質．デフォルトは99．
- `--model_name`：認識率を測るモデルの指定．
- `--gpu`：どのGPUを使用するか指定．
- `--crf`：圧縮した動画を検証時に使う場合指定．（圧縮した時のCRFのオプション，デフォルト−10（元動画））
- `--gop`：圧縮した動画を検証時に使う場合指定．（圧縮した時のGOPのオプション，デフォルト−10（元動画））

#### compression_train.py
- `--dataset`：datasetの名前．例：Kinetics400,  UCF101
- `--train_data_path`：学習に使用するデータセットのパスを指定．デフォルトは`dataset_imgaug.py`と同じ．
- `--train_crf`：圧縮した動画を学習時に使う場合指定．（圧縮した時のCRFのオプション，デフォルト−10（元動画））
- `--train_gop`：圧縮した動画を学習時に使う場合指定．（圧縮した時のGOPのオプション，デフォルト−10（元動画））
- `--val_data_path`：検証に使用するデータセットのパスを指定．デフォルトは`dataset_imgaug.py`と同じ．
- `--val_crf`：圧縮した動画を検証時に使う場合指定．（圧縮した時のCRFのオプション，デフォルト−10（元動画））
- `--val_gop`：圧縮した動画を検証時に使う場合指定．（圧縮した時のGOPのオプション，デフォルト−10（元動画））
- `--model_name`：認識率を測るモデルの指定．
- `--gpu`：どのGPUを使用するか指定．
- `--model_pth`：学習したモデルを保存するパスを指定．model_path/`dataset`/`model_name`/`model_pth`.pth
  - デフォルトは元動画の`original`
- `--batch_size`：デフォルトはconfig.iniで指定．
- `--epoch`：デフォルトはconfig.iniで指定．
  
#### model_load_val.py
- `--dataset`：datasetの名前．例：Kinetics400,  UCF101
- `--path`：ロードするデータセットのパスを指定．デフォルト以下のようなる．
  - Kinetics400 : /mnt/dataset/Kinetics400/
  - UCF101 : /mnt/dataset/UCF101/
- `--use_compression`：JPEG圧縮をかけるかどうか．圧縮をかける場合はオプションが必要．
- `--compression_strength`：JPEG圧縮の圧縮強度の指定[0 ~ 100]．0が高画質で100が低画質．デフォルトは99．
- `--model_name`：認識率を測るモデルの指定．
- `--gpu`：どのGPUを使用するか指定．
- `--crf`：圧縮した動画を検証時に使う場合指定．（圧縮した時のCRFのオプション，デフォルト−10（元動画））
- `--gop`：圧縮した動画を検証時に使う場合指定．（圧縮した時のGOPのオプション，デフォルト−10（元動画））
- `--mod_pth`：検証時に使用する保存したモデルのパスを指定．
  - デフォルトは元動画で学習した時の`original_best.pth`
  


