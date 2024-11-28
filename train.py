import os
import pandas as pd
import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F

from torch_local.model import Model
from torch_local.losses import CTCLoss
from torch_local.dataProvider import DataProvider
from torch_local.metrics import CERMetric, WERMetric
from torch_local.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay

# from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from mltu.preprocessors import AudioReader
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding
from configs import ModelConfigs


configs = ModelConfigs()

# Dataset Path
dataset_path = "Datasets/ahshanti_wav"
metadata_path = os.path.join(dataset_path, "data.csv")
wavs_path = os.path.join(dataset_path, "wavs")


# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="\t", header=None, quoting=3)
dataset = []
vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e','ɛ', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ɔ','p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


metadata_df['Audio Filepath'] = metadata_df['Audio Filepath'].str.replace('.ogg', '.wav')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in metadata_df.columns:
    metadata_df.drop(columns=['Unnamed: 0'], inplace=True)


for file_name, transcription, _ in metadata_df.values.tolist():
    path = f"Datasets/ashanti_wav/wavs/{file_name}.wav"
    new_label = "".join([l for l in transcription.lower() if l in vocab])
    dataset.append([path, new_label])

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=16000),
        ],
    transformers=[
        LabelIndexer(vocab),],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(vocab), use_on_batch=True),
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=configs.train_workers,
)

train_dataProvider, test_dataProvider = data_provider.split(split=0.9)

# train_dataProvider.augmentors = [
#         RandomAudioNoise(), 
#         RandomAudioPitchShift(), 
#         RandomAudioTimeStretch()
#     ]

vocab = sorted(vocab)
configs.vocab = vocab
configs.save()

class CustomWav2Vec2Model(nn.Module):
    def __init__(self, hidden_states, dropout_rate=0.2, **kwargs):
        super(CustomWav2Vec2Model, self).__init__( **kwargs)
        pretrained_name = "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_name, vocab_size=hidden_states, ignore_mismatched_sizes=True)
        self.model.freeze_feature_encoder() # this part does not need to be fine-tuned

    def forward(self, inputs):
        output = self.model(inputs, attention_mask=None).logits
        # Apply softmax
        output = F.log_softmax(output, -1)
        return output

custom_model = CustomWav2Vec2Model(hidden_states = len(vocab)+1)

# put on cuda device if available
if torch.cuda.is_available():
    custom_model = custom_model.cuda()

# create callbacks
warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    final_lr=configs.final_lr,
    initial_lr=configs.init_lr,
    verbose=True,
)

tb_callback = TensorBoard(configs.model_path + "/logs")

earlyStopping = EarlyStopping(monitor="val_CER", patience=16, mode="min", verbose=1)

modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)

model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.max_audio_length), 
    verbose=1,
    metadata={"vocab": configs.vocab},
    dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}}
)

# create model object that will handle training and testing of the network
model = Model(
    custom_model, 
    loss = CTCLoss(blank=len(configs.vocab), zero_infinity=True),
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay),
    metrics=[
        CERMetric(configs.vocab), 
        WERMetric(configs.vocab)
    ],
    mixed_precision=configs.mixed_precision,
)

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))

model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[
        warmupCosineDecay, 
        tb_callback, 
        earlyStopping,
        modelCheckpoint, 
        model2onnx
    ]
)