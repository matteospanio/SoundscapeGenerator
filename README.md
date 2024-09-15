# SoundscapeGenerator
A project to use Stable Diffusion to generate soundscapes with emotion tags as input.

# Dependencies
- For setup you need to create a conda environment, if you don't want the whole conda package you can get a minified version from [this page](https://docs.anaconda.com/miniconda/miniconda-install/).
- Create a conda virtual environment
  ```bash
  conda create -n my_env python=3.9
  ```
- Activate/Deactivate the environment
  ```bash
  activate my_env # activates the env
  deactivate my_env # exits the env
  ```
- Install the dependencies:
  ```bash
  bash scripts/conda_env_deps.sh
  ```

  > The default settings have been tested on linux. You could need to modify this script according to your OS. Visit https://pytorch.org/ and formulate your conda command based on your OS, such as:
  >   - OSX: `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
  >   - Windows (check your CUDA version): `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

# Setup
- To begin with activate your env and run the script preparation.py to download the required DEAM dataset.
  - `python preparation.py`
- Create the spectrograms
  - `python create_spectrograms.py`
- Prepare the dataset
  - `python prepare_dataset.py`
- Categorize the spectrograms based on their emotion classes
  - `python categorize_spectrograms.py`
- (Optional) If you want to generate K-means clustering map
  - `python cluster_emotions.py`
## References

- Used the helper functions from, great help:
  - https://github.com/chavinlo/riffusion-manipulation.git