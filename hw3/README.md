# Installation

You can use basically the same environment as in hw2, with just one modification.

```bash
conda activate <hw2-env-name>
pip install dotmap==1.3.30 gymnasium[box2d]==0.27.1
```

if you are using zsh as your shell, use the following command instead:

```zsh
conda activate <hw2-env-name>
pip install dotmap==1.3.30 gymnasium\[box2d\]==0.27.1
```

We use the `dotmap` package to have a "dot-able" config dictionary as a substitution for the default one, as we find the original dictionary from hydra is slow.

You may encounter several errors when installing `gynmasium[box2d]` depending on your system and requirements installed previously, here's an incomplete list of how to get over them:

1. `error: command 'swig.exe' failed: None` or `command 'swig' failed: No such file or directory`

   In that case you can install `swig` via `conda install swig` in your `<hw2-env-name>` environment and try again. If the same error persists, try the guide in this [link](https://open-box.readthedocs.io/en/latest/installation/install_swig.html) to install `swig` manually.

2. On the window platform, you may encounter the problem:`error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"` 
   
   In that case, you can follow this [link](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/) the error message provided, download the build tool file, and run it. You'll need to select the "Desktop development with C++" checkbox in the "Workloads", and you may remove the optional requirements in the right sidebar (MSVC, Windows SDK, CMake tools, etc.) 
   
   After the installation is finished, restart your computer and try installing `gymnasium[box2d]` again (if you encounter the same error, try to select the optional dependencies and try again). There may be other issues concerning the installation of `gymnasium[box2d]`, please contact us if you find yourself in a different situation.

That's it! If you encounter any trouble creating the environment, please let us know :-)