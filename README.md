#Usage

## direct call

**session 1**

```
$ CUDA_VISIBLE_DEVICES=1,3 python remote_train.py --rank 0 --world_size 2
```
    
**session 2**

```
$ CUDA_VISIBLE_DEVICES=1,3 python remote_train.py --rank 1 --world_size 2
```

---

## run with Docker

Note: nvidia-docker has to be installed

```
$ docker build --file Dockerfile --tag mini .
$ CUDA_VISIBLE_DEVICES=1,3 docker-compose --file docker-compose-m2.yml up
```

    