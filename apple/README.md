# MLX-TADA

TADA speech synthesis on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Setup

```bash
pip install -e ".[convert]"
```

For auto-transcription of reference audio (optional):
```bash
pip install mlx-whisper
```

## Convert Weights

```bash
# 3B model
python -m mlx_tada.convert_3b ./mlx_weights

# 1B model
python -m mlx_tada.convert_1b ./mlx_weights
```

## Generate Speech

### CLI

```bash
python -m mlx_tada.generate \
  --weights ./mlx_weights \
  --audio speaker.wav \
  --text "Hello world, today is a nice day." \
  --output output.wav
```

With 4-bit quantization (10x faster):
```bash
python -m mlx_tada.generate \
  --weights ./mlx_weights \
  --audio speaker.wav \
  --text "Hello world, today is a nice day." \
  --quantize 4 \
  --output output.wav
```

### Python

```python
from mlx_tada import TadaForCausalLM

model = TadaForCausalLM.from_weights("./mlx_weights", quantize=4)
ref = model.load_reference("speaker.wav")
out = model.generate("Hello world, today is a nice day.", ref)

# out.audio     - numpy float32 array (24kHz)
# out.duration  - audio duration in seconds
# out.rtf       - real-time factor
# out.num_tokens
```

Save and reuse references:
```python
from mlx_tada import Reference

ref = model.load_reference("speaker.wav")
ref.save("speaker.npz")

ref = Reference.load("speaker.npz")
out = model.generate("Reusing the same voice.", ref)
```

Save audio:
```python
from mlx_tada import save_wav
save_wav(out.audio, "output.wav")
```

## Debug Logging

```bash
DEBUG=1 python -m mlx_tada.generate --weights ./mlx_weights --audio speaker.wav --text "Hello"
```

```python
from mlx_tada.model import setup_logging
setup_logging()
```
