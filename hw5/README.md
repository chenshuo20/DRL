# Dataset
For your convenience, we've downloaded all the datasets you may need for you such that you don't need to download from the D4RL library, which could be unstable.

The datasets can be found at this [cloud link](https://cloud.tsinghua.edu.cn/d/18abd6fda20b4b47ba65/). For our homework, you just need to download the three `medium` datasets, but we provided all datasets if you're interested or doing the bonus part.

After downloading the datasets, please don't change their name and put them into the `dataset` folder of the root directory.
# Environment
You can use basically the same environment as in hw4, with just one modification:

```bash
conda activate <hw4-env-name>
pip install gymnasium[mujoco]==0.27.1
```
Rollback the MuJoCo package to version 2.3.3 if you encounter the following error:
"XML Error: global coordinates no longer supported. To convert existing models, load and save them in MuJoCo 2.3.3 or older"
```bash
pip install mujoco==2.3.3
```

if you are using zsh as your shell, use the following command instead:

```zsh
conda activate <hw4-env-name>
pip install gymnasium\[mujoco\]==0.27.1
```

A successful installation of the MuJoCo package could be troublesome, but hopefully, the situation has improved a lot since the shift from the old `mujoco_py` packages. If you encounter problems installing the package, first refer to their GitHub [repo](https://github.com/deepmind/mujoco) for guidance.

After successfully installing the MuJoCo library and its dependencies, it's safe for you to claim yourself an "experienced" RL researcher :-)