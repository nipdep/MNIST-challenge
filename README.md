# MNIST-challenge
 _codebases for kaggle MNIST challenge_ \

## environment building
 * for tensorflow framework
 ```
 pip install -r requirements.txt
 ```

 * for Pytorch framework
 ```
 conda install pytorch
 ```
 * by docker container [only support Pytorch framework]
    - in vscode \
    by using docker plugin build docker container from the Dockerfile
    - in terminal \
    `Docker build -f Dockerfile -t MNIST-challenge .`
 * by Kaggle Dockerfile [ for all the frameworks] \
    [Dockerfile link](https://github.com/Kaggle/docker-python/blob/master/Dockerfile)


