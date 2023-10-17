# Installation

Since we are using PyTorch for hw2, we recommend using conda to manage the environment. Please refer to the [miniconda](https://docs.conda.io/en/latest/miniconda.html) homepage for a compact conda installation.

You have two options for creating the environment of hw2

* For mac users or a cpu-only installation, please remove the `pytorch-cuda` term in either ways.
* To create a new conda environment, simply run `conda env create -f environment.yml`
* If you want to install the package within the environment you created with hw1, please following the below steps:

  ```bash
  conda activate <hw1-env-name>
  # we are using PyTorch 2.0!
  # remove the pytorch-cuda=11.7 term if you are a mac user to want a cpu-only installation
  conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install gymnasium[classic_control]==0.27.1
  pip install matplotlib==3.7.1
  # for hyperparameter management
  pip install hydra-core==1.3.2
  # for video recording
  pip install moviepy==1.0.3
  ```

That's it! If you encounter any trouble creating the environment, please let us know :-)
