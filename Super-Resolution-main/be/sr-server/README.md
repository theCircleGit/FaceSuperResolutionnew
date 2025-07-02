If you encounter the error 'RuntimeError: Detected that PyTorch and torchvision were compiled with different CUDA versions. PyTorch has CUDA Version=11.8 and torchvision has CUDA Version=11.7. Please reinstall the torchvision that matches your PyTorch install.
', uninstall torchvision and reinstall with the following commands (you may need to install onnx gpu runtime too):

```
pip uninstall torchvision
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

python3 -m pip install onnxruntime-gpu==1.18.0
```


Run celery worker with:
```
python -m celery -A sr_api worker --loglevel=debug --concurrency=10
```
