# End to End Personalize Image Enhancement
PIE task 를 수행하는 End to End 모델 연구

기존의 PIE task 에서는 End to End 구조를 가진 Model의 성능이 저조했다. 많은 모델들이 Style Learning 후 Enhancement를 진행하는 2-step 단계로 학습을 진행하였다.
하지만 이런 2-step Learning에서는 학습된 Style-Parameter가 직접적으로 PIE task의 성능 강화로 이어진다는 명확한 근거가 부족하다. 따라서 이를 개선하고자 End to End로 학습하는 모델을 연구하였다.
Style Enocoder 와 Enhancement Model을 연결하여 두 모델을 여러가지 다른 모델들을 참고해 수정하고, 독자적인 아이디어를 통해 수정하는 방식으로 진행하였다. 

# Structure
- AdaIN : Adaptive Interval Normalization
- Enahncer : StarEnhancer(StarEnhancer), UEnhancer(U-net), MyEnhancer(Attention-based), AdaINPieNet(Pienet)
- Style Encoder : pSp(pSpGAN Encoder)


### Folder

```
┬─ data
│    ├─ Adobe5K
│    │   ├─ train
│    │   │   ├─ a
│    │   │   │  ├─ a0001.jpg
│    │   │   │  ├─ ...(img)
│    │   │   │  └─ a4500.jpg
│    │   │   ├─ b
│    │   │   ├─ ...        
│    │   │   ├─ e    
│    │   │   └─ raw
│    │   └─ test
│    └─ Preset
│        ├─ train
│        └─ test
├─ dataset
│    └─ dataloader.py
├─ imgsave
│    └─ enahncer / gt / pref / raw / style folder --- training img save
├─ Logs
│    └─ Best / Train / Valid Log.txt
└─ train.py
```

### Dataset
1. MIT - Adobe5K (Expert A~E)
2. Adobe Lightroom Prest 1~13

### Enviroment

```sh
conda create -n test python=3.7.5
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install opencv
conda install numpy==1.21.5
conda install -c conda-forge matplotlib scikit-learn
conda install -c conda-forge tqdm
```

### Train
```sh
python train.py
```

### Reference
- [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
- [StarEnhancer](https://github.com/IDKiro/StarEnhancer)
- [PieNet](https://github.com/hukim1124/PieNet)
- [StarEnhancer](https://github.com/IDKiro/StarEnhancer)
- [Maskesd-style-Modeling](https://github.com/satoshi-kosugi/masked-style-modeling)
- iUP-Enhancer
