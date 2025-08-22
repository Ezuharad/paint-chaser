# pAInt chaser
A reinforcement learning agent that learns to play UFO50's Paint Chase using [REINFORCE](https://arxiv.org/abs/2010.11364).

## Overview
pAInt chaser is a project that demonstrates the application of reinforcement learning to video game playing. The agent learns to play the Paint Chase minigame from UFO50 by using the REINFORCE algorithm, a fundamental policy gradient method in reinforcement learning.

The agent uses a convolutional neural network to directly process screen pixels, with a residual network architecture for better gradient flow and learning capability.

### Features
- Direct pixel-to-action learning using deep neural networks
- REINFORCE algorithm implementation with batched updates
- Configurable training parameters via environment variables and CLI arguments
- Real-time training metrics and model checkpointing
- Screen capture and preprocessing pipeline

## Installation
### Prerequisites
- Python 3.13 or higher. 3.11 and higher may work, but has not been tested.
- [UFO50](http://50games.fun/), which can be purchased on Steam.

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ezuharad/paint-chase.git
cd paint-chase
```

2. Create and activate a virtual environment. Python's native environment management utilities (venv and pip), as well as [uv](https://docs.astral.sh/uv/), should work.
```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the package.
```bash
uv pip install -r requirements.txt
```

## Usage
### Training
The agent can be trained using the provided [training script](script/train_reinforce.py). The most basic usage is as follows:

```bash
python3 script/train_reinforce.py <BATCHES> --model-out <MODEL-OUTPUT-PATH>
```

In order for training to work, Paint Chase must be running and in focus, with the game's screen region completely covered by the dimensions specified in the `SCREEN_REGION` constant defined in [train_reinforce.py](script/train_reinforce.py) in the top left corner of your desktop. The currently defined constant is designed for the minimum game resolution. Because the neural network was designed with the lowest resolution available, changing the dimensions of the region aren't recommended, although it is possible to just change the resblock size.

### Configuration
Training parameters can be configured through environment variables or command-line arguments. CLI arguments take precedence over environment variables.

As an example, the environment variables:
```bash
export PAINT_CHASER_LEARNING_RATE=0.001
export PAINT_CHASER_BATCH_SIZE=8
python3 script/train_reinforce.py 8 \
    --model-output a.pt
```

are equivalent to the command line arguments:
```bash
python3 script/train_reinforce.py 8 \
    --model-output a.pt \
    --learning-rate 0.001 \
    --batch-size 8 \
```

Available parameters are given below:


| Parameter | Environment Variable | CLI Argument | Default | Description |
|-----------|---------------------|--------------|---------|-------------|
| Number of Batches | None | `batches` | None | Number of batches to train. **Required positional argument** |
| Output Model | None | --model-out | None | Path to save model weights to. **Required argument** |
| Batch Size | PAINT_CHASER_BATCH_SIZE | --batch-size | 1 | Number of episodes to train in one batch |
| Learning Rate | `PAINT_CHASER_LEARNING_RATE` | `--learning-rate` | 1e-5 | Optimizer learning rate |
| Save Every N Batches | `PAINT_CHASER_SAVE_EVERY` | `--save-every-n-batch` | 10 | Number of batches to wait between checkpoint saves |
| Gamma Good | `PAINT_CHASER_GAMMA_GOOD` | `--gamma-good` | 0.95 | Reward discount used for the number of blue pixels on the screen |
| Gamma Bad | `PAINT_CHASER_GAMMA_BAD` | `--gamma-bad` | 0.99| Penalty discount used for the number of red pixels on the screen |
| Bad Penalty Factor | `PAINT_CHASER_BAD_PENALTY_FACTOR` | --bad-penalty-factor | 0.3 | Coefficient for weighting penalty from the number of red pixels on the screen |
| Input Model | None | --model-in | None | Pretrained weights to use |
| Log Path | None | --log-out | /dev/null | File to print log outputs to |
| Use CPU | PAINT_CHASER_USE_CPU | --use-cpu | None | Switch for using the CPU instead of CUDA or MPS |

## License
This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.

## Acknowledgments
- Mossmouth, for creating UFO 50 and Paint Chase
- The PyTorch team
- OpenAI, as well as the team responsible for gymnasium

