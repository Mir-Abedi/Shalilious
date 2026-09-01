#!/usr/bin/env bash
echo "host=$(hostname)  pwd=$(pwd)  user=$(id -un)"
echo "--- repo visible from inside the container?"
ls src/ 2>&1 | head
echo "--- dataset present?"
ls -la data/ 2>&1 | tail -3
echo "--- python"
which python3; python3 -V
python3 -c "import torch, torchvision, numpy; print('torch', torch.__version__, 'tv', torchvision.__version__, 'np', numpy.__version__)"
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "--- network reachable?"
curl -sI --max-time 15 https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | head -1 || echo "NO NETWORK"
echo "PROBE_DONE"
