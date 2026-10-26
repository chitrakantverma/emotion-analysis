# Expression model

## Pretrained FER+ model

- File: emotion-ferplus-8.onnx
- Source: ONNX Model Zoo, FER+ Emotion Recognition
- URL: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus
- Weight URL: https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
- The model-specific upstream README identifies MIT (see UPSTREAM_README.md).
  FERPLUS_MIT_LICENSE.txt preserves the original FERPlus MIT notice.
  BASELINE_LICENSE.txt preserves the ONNX Model Zoo repository's Apache-2.0
  licence. These notices cover the upstream model and repository.
- Paper: E. Barsoum, C. Zhang, C. C. Ferrer and Z. Zhang, *Training Deep Networks
  for Facial Expression Recognition with Crowd-Sourced Label Distribution*,
  ICMI 2016. https://arxiv.org/abs/1608.01041
- Upstream training: CNTK, FER+ annotation distribution, cross-entropy mode.
- Input: raw grayscale 0–255, 64×64, float32, NCHW (1,1,64,64).
- Output: eight logits, followed by softmax in our code.
- Order: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.
- Project UI synonyms: happiness→happy, sadness→sad, anger→angry,
  surprise→surprised, fear→fearful. Synonyms do not change class meaning.
- Unsupported winners: disgust/contempt are shown as outside scope. Original
  eight-class probability distribution is preserved.
- No accuracy result on the user's own webcam or test set has been established.

## Training a six-class model

`train.py` defines a six-class CNN trained separately from the included model.
48×48 grayscale input, 0–1 normalisation, NHWC, six softmax probabilities.
Order: angry, fearful, happy, sad, surprised, neutral.
It requires FER-2013 data and a training run. The loader checks that the model
metadata matches its input format and class order.

## Known limits

No internal emotional-state inference; no demographic fairness guarantee;
no medical, behavioural or identity claims. Natural webcam performance can
differ from dataset results. Single-frame predictions can fluctuate. No
confidence calibration or new research novelty is claimed.
