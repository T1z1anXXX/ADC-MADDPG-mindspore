# ADC-MADDPG-mindspore
## 简介
This study focuses on the resilient path planning of a data collection unmanned aerial vehicle (UAV) swarm within a scenario that efficiently gathers information from Internet of Things (IoT) nodes. In this configuration, the UAV swarm confronts the challenge posed by intelligent jamming adversaries, which possess the ability to adapt and learn, thereby sending jamming signals in proximity to the swarm and degrading the communication quality. To counteract the aforementioned intelligent jamming attacks, the UAV swarm utilizes a path planning algorithm that not only ensures resilience but also facilitates the efficient collection of data from diverse IoT nodes. This is accomplished by taking into account various constraints, including kinematic limitations, airspace restrictions, and mission-specific deadlines. Inspired by the multi-agent deep deterministic policy gradient (MADDPG) algorithm, this work puts forward a resilient reinforcement learning scheme for the UAV swarm. The simulation part thoroughly evaluates the resilience of different defensive algorithms under diverse conditions, encompassing scenarios without jamming attacks, those with fixed attacks, and those with intelligent jamming attacks. Furthermore, the work proposes a suite of advanced algorithms derived from the MADDPG algorithm: attention-critic-MADDPG (AC-MADDPG), dual-critic-MADDPG (DC-MADDPG), and attention-dual-critic-MADDPG (ADC-MADDPG), all of which are designed to enhance resilience against jamming attacks.
## 使用说明
1. 训练 main_openai.py

2. 测试 test.py

3. 参数

  ·主要参数1 --jammer_act store_true类型，添加该参数将会在训练过程中允许干扰机运动

  ·主要参数2 --train_model (m: MADDPG, a: AC-MADDPG, d: DC-MADDPG, adc: ADC-MADDPG)
