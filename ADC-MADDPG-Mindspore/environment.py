TIMESTAMP=0.05#时间步长
TIME=0#时间步
α=1.5#路径损耗指数
TSV=3#信噪比阈值
A_MAX=10
V_MAX=5
X_MAX=10
R_MAX=10000
D=0.2

import numpy as np
import math
import threading
import copy
import heapq
from itertools import chain
import time

PV_INIT=[
    np.array(([0,0,10]),dtype=np.float32),
    np.array(([0,5,10]),dtype=np.float32),
    np.array(([5,0,10]),dtype=np.float32),
    np.array(([7,3,5]),dtype=np.float32),
    np.array(([4,9,5]),dtype=np.float32)
]
DATA_INIT=[8.5,8.6,8.8,8.4,7.2,8.1,8.3,8.4,8.4,9.3,9.4,10.4]

def dis(p1,p2,h):#三维距离
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + h**2)**0.5

def relative_coordinates(p1,p2): #计算两台无人机的相对横纵坐标(标准化)
    return np.array([p2[0]-p1[0],p2[1]-p1[1]])/X_MAX

def get_p_recv (pV1,pV2):#计算两个坐标之间的通信系数（我自己起的名字，反正这个系数乘上发送功率能得到实际功率）
    l = ((pV1[0] - pV2[0]) ** 2 + (pV1[1] - pV2[1]) ** 2 + (pV1[2] - pV2[2]) ** 2) ** (α / 2)  # 路径损耗
    g = abs(pV1[2]-pV2[2]) / ((pV1[0] - pV2[0]) ** 2 + (pV1[1] - pV2[1]) ** 2 + (pV1[2] - pV2[2]) ** 2) ** 0.5#天线增益
    return g/l
class ENVIRONMENT:
    def __init__(self,size,UAVs,JAMMERs,IoTNodes,NFZones):
        self.size=size
        self.UAVs=UAVs
        self.JAMMERs=JAMMERs
        self.IoTNodes=IoTNodes#物联网节点
        self.NFZones=NFZones#禁飞区 格式为np.array([x_min,x_max,y_min,y_max])
        self.UAVthreads=[]#无人机的线程池
        self.JAMMERthreads = []#干扰机的线程池
        self.done_cnt=0#物联网节点传输完成计数
        self.dis_martrix=np.zeros((len(self.UAVs),len(self.UAVs)),dtype=np.float32)#保存无人机之间的距离的矩阵
        self.observation_space=[len(uav.normalization_s(environment=self)) for uav in chain(self.UAVs,self.JAMMERs)]
        self.action_space=[2 for uav in  chain(self.UAVs,self.JAMMERs)]
        self.n=len(self.UAVs)+len(self.JAMMERs)


    def reset(self, jammer=True, random=False):
        self.done_cnt = 0
        for uav,pV in zip(chain(self.UAVs,self.JAMMERs),PV_INIT):
            uav.reset(pV=pV,environment=self)
        for node,data in zip(self.IoTNodes,DATA_INIT):
            node.data=data
            node.done=False
            node.connected=False
        #self.uav_observe(timestep=0)
        for i,uav in enumerate(chain(self.UAVs,self.JAMMERs)):
            if i == 0:
                s = [list(uav.normalization_s(environment=self))]
            else:
                s.append(list(uav.normalization_s(environment=self)))
            if i >=3 and not jammer:
                self.JAMMERs[i-3].pJ=0
        if random:
            self.random_iotnodes()
        return s

    def random_iotnodes(self):
        np.random.seed(int(time.time()))
        node_test_0 = IoTNode(x=1.8 + np.random.randint(-2, 2), y=5 + np.random.randint(-2, 2), p=0.01, data=8.5)
        node_test_1 = IoTNode(x=2.5 + np.random.randint(-2, 2), y=4.8 + np.random.randint(-2, 2), p=0.01, data=8.6)
        node_test_2 = IoTNode(x=8.3 + np.random.randint(-2, 2), y=7 + np.random.randint(-2, 2), p=0.01, data=5.8)
        node_test_3 = IoTNode(x=2.7 + np.random.randint(-2, 2), y=4.6 + np.random.randint(-2, 2), p=0.01, data=8.4)
        node_test_4 = IoTNode(x=5.3 + np.random.randint(-2, 2), y=7.5 + np.random.randint(-2, 2), p=0.01, data=7.2)
        node_test_5 = IoTNode(x=8 + np.random.randint(-2, 2), y=5 + np.random.randint(-2, 2), p=0.01, data=9.1)
        node_test_6 = IoTNode(x=6.6 + np.random.randint(-2, 2), y=2.7 + np.random.randint(-2, 2), p=0.01, data=9.3)
        node_test_7 = IoTNode(x=6 + np.random.randint(-2, 2), y=3.2 + np.random.randint(-2, 2), p=0.01, data=8.4)
        node_test_8 = IoTNode(x=7 + np.random.randint(-2, 2), y=4.2 + np.random.randint(-2, 2), p=0.01, data=8.4)
        node_test_9 = IoTNode(x=6.6 + np.random.randint(-2, 2), y=6.7 + np.random.randint(-2, 2), p=0.01, data=9.3)
        node_test_10 = IoTNode(x=4 + np.random.randint(-2, 2), y=3.2 + np.random.randint(-2, 2), p=0.01, data=9.4)
        node_test_11 = IoTNode(x=2.8 + np.random.randint(-2, 2), y=2.2 + np.random.randint(-2, 2), p=0.01, data=10.4)
        self.IoTNodes=[node_test_0, node_test_1, node_test_2, node_test_3, node_test_4,
                                             node_test_5, node_test_6, node_test_7, node_test_8, node_test_9,
                                             node_test_10, node_test_11]
    def uav_observe(self,timestep):#为所有的典型机并发执行observe函数
        self.UAVthreads = []
        for uav in chain(self.UAVs,self.JAMMERs):
            self.UAVthreads.append(threading.Thread(target=uav.observe,args=(self,timestep)))
        for t in self.UAVthreads:
            t.start()
            t.join()

    def step(self,timestep,action_n,jammer_act=False):
        self.uav_observe(timestep=timestep)
        reward_n=[]
        done_n=[]
        for i in range(3):
            for j in range(i+1,3):
                self.dis_martrix[i][j] = self.dis_martrix[j][i] = dis(self.UAVs[i].pV,self.UAVs[j].pV,0)
        total_reward=0
        if jammer_act:
            for i,uav in enumerate(chain(self.UAVs,self.JAMMERs)):
                reward_n.append(uav.step(environment=self, timestep=timestep,action=action_n[i]))
                total_reward += reward_n[i]
                done_n.append(uav.arrive) if i < 3 else done_n.append(False)
                if i == 0:
                    new_s = [list(uav.normalization_s(environment=self))]
                else:
                    new_s.append(list(uav.normalization_s(environment=self)))
            # reward_t_n = np.mean(reward_n[:3])
            # reward_j_n = np.mean(reward_n[-2:])
            # reward_mean_n =[reward_t_n for _ in range(len(self.UAVs))]
            # reward_mean_n += [reward_j_n] *len(self.JAMMERs)
            return new_s,reward_n,done_n
        else:
            for i,uav in enumerate(chain(self.UAVs,self.JAMMERs)):
                if i < 3:
                    reward_n.append(uav.step(environment=self, timestep=timestep,action=action_n[i]))
                else:
                    reward_n.append(uav.step(environment=self, timestep=timestep, action=[0,0]))
                total_reward += reward_n[i]
                done_n.append(uav.arrive) if i < 3 else done_n.append(False)
                if i == 0:
                    new_s = [list(uav.normalization_s(environment=self))]
                else:
                    new_s.append(list(uav.normalization_s(environment=self)))
            #reward_t_n = np.mean(reward_n[:3])
            #reward_j_n = np.mean(reward_n[-2:])
            #reward_mean_n =[reward_t_n for _ in range(len(self.UAVs))]
            #reward_mean_n += [reward_j_n] *len(self.JAMMERs)
            return new_s,reward_n,done_n

class UAV:#无人机基类
    # # 将变量拼接起来得到状态
    # def combine(self):
    #     self.s = np.concatenate((self.pV, self.vV, self.r, self.vM))

    def __init__(self,pV,vV,r,vM,num):
        self.pV=pV #当前位置 三维坐标
        self.vV=vV #当前速度 x和 y方向
        self.r=r  #建模的半径
        self.vM=vM #速度最大值
        self.num=num#编号


    # def action(self,a,environment):#根据动作进行状态转移计算 动作action定义为[x方向加速度,y方向加速度]的np数组
    #     coll_cnt=0#出界的维度计数量
    #     self.pV[0]  =  self.pV[0] + self.vV[0] * TIMESTAMP + 0.5 * a[0] * TIMESTAMP**2
    #     self.pV[1]  =  self.pV[1] + self.vV[1] * TIMESTAMP + 0.5 * a[1] * TIMESTAMP**2
    #     self.vV[0] += a[0] * TIMESTAMP
    #     self.vV[1] += a[1] * TIMESTAMP
    #     # 当和边界发生碰撞 速度置为反向
    #     if self.pV[0] < 0 or self.pV[0] > environment.size[0]:
    #         coll_cnt += 1
    #     if self.pV[1] < 0 or self.pV[1] > environment.size[1]:
    #         coll_cnt += 1
    #     return coll_cnt

class JAMMER(UAV):#干扰机
    def __init__(self,pV,vV,r,vM,num,pJ):
        super(JAMMER, self).__init__(pV,vV,r,vM,num)
        self.pJ=pJ #干扰功率
        self.target_t=-1#尝试干扰的目标典型机
        self.target_rnv=-1#目标典型机的信噪比
        self.target_d = -1

    def reset(self,pV,environment):
        self.pV=copy.deepcopy(pV)
        self.vV=np.array([0,0],dtype=np.float32)
        self.target_t=-1#尝试干扰的目标典型机
        self.target_rnv=-1#目标典型机的信噪比
        self.target_d=-1

    def normalization_s(self,environment):
        return np.concatenate((self.pV[0:2] / X_MAX,
                               self.vV / V_MAX,
                               relative_coordinates(self.pV, environment.UAVs[self.target_t].pV) if self.target_t != -1 else np.array([0,0],dtype=np.float32) ,
                               np.array([self.target_d / X_MAX],dtype=np.float32),
                               np.array([self.target_rnv],dtype=np.float32)),
                              axis=0)

    def observe(self,environment,timestep):#观察环境，选择一个目标典型机
        #贪心策略，选择距离最近的无人机
        d_temp=dis(environment.UAVs[0].pV,self.pV,0)
        self.target_d=d_temp
        self.target_t=0
        self.target_rnv=environment.UAVs[0].rnv
        for i in range(1,3):
            d_cur=dis(environment.UAVs[i].pV,self.pV,0)
            if  d_cur < d_temp:
                d_temp = d_cur
                self.target_d = d_temp
                self.target_t=i
                self.target_rnv=environment.UAVs[i].rnv

    def action(self,a):#根据动作进行状态转移计算 动作action定义为[x方向加速度,y方向加速度]的np数组
        coll_flag=0#出界计数
        self.pV[0]  =  self.pV[0] + self.vV[0] * TIMESTAMP + 0.5 * a[0] * TIMESTAMP**2
        self.pV[1]  =  self.pV[1] + self.vV[1] * TIMESTAMP + 0.5 * a[1] * TIMESTAMP**2
        self.vV[0]  =  float(np.clip(self.vV[0] + a[0] * TIMESTAMP,-V_MAX,V_MAX))
        self.vV[1]  =  float(np.clip(self.vV[1] + a[1] * TIMESTAMP,-V_MAX,V_MAX))
        # 出界了
        if self.pV[0] <= 0 or self.pV[0]  >= X_MAX:
            coll_flag += 1
        if self.pV[1] <= 0 or self.pV[1]  >= X_MAX:
            coll_flag += 1
        return coll_flag

    def step(self,action,environment,timestep):
        #print(action)
        r=0
        r -= 200*self.action(a=action*A_MAX)#出界惩罚
        self.observe(environment,timestep=timestep)
        #信噪比分段奖励项
        if not environment.UAVs[self.target_t].connected:
            r += 1000
        else:
            if self.target_rnv >= 3.5:
                r -= self.target_rnv*200
            elif 0 < self.target_rnv < 3.5:
                r += 500/self.target_rnv
        #和当前目标距离的分段奖励项
        if self.target_d >= 5:
            r -= self.target_d * 200
        elif 0 < self.target_d < 5:
            r += 2000/(10+self.target_d)
        r = np.clip(r, -R_MAX, R_MAX)
        return r / R_MAX

class TYPICAL(UAV):  #典型机
    def __init__(self,pV,vV,r,vM,num,pD):
        super(TYPICAL, self).__init__(pV,vV,r,vM,num)
        self.connected=False
        self.pD = pD  #目的地址
        self.rnv=0#有效信息率
        self.connected_node=-1#保存连接节点的下标
        self.data=0#在一个时间间隔内收集到的数据量
        self.arrive=False#是否到达终点
        self.jammer_rep = np.array([0, 0], dtype=np.float32)  # 干扰机造成的排斥力
        self.close_2 = []#最近的两台无人机的编号
        # self.jammer_pos_q = [queue.Queue(5),queue.Queue(5)]#保存干扰机过去5个时间步位置信息的队列
    def reset(self,pV,environment):
        self.pV=copy.deepcopy(pV)
        self.vV=np.array([0,0],dtype=np.float32)
        self.rnv=0#有效信息率
        self.connected_node=-1#保存连接节点的下标
        self.conected=False
        self.data=0#在一个时间间隔内收集到的数据量
        self.arrive=False#是否到达终点
        self.jammer_rep=np.array([0,0],dtype=np.float32)#干扰机造成的排斥力
        self.close_2=[]
        # for i,q in enumerate(self.jammer_pos_q):
        #     q.queue.clear()
        #     for j in range(5):
        #         q.put(copy.deepcopy(environment.JAMMERs[i].pV))
    def normalization_s(self,environment):
        self.close_2 = list(map(list(environment.dis_martrix[self.num]).index, heapq.nsmallest(3, environment.dis_martrix[self.num])))[1:]
        return np.concatenate((self.pV[0:2]/X_MAX,
                               self.vV/V_MAX,
                               relative_coordinates(self.pV, environment.UAVs[self.close_2[0]].pV),
                               relative_coordinates(self.pV, environment.UAVs[self.close_2[1]].pV),
                               np.array([1] if self.connected else [-1]),
                               relative_coordinates(self.pV, environment.JAMMERs[0].pV),
                               relative_coordinates(self.pV, environment.JAMMERs[1].pV),
                               np.array( [(environment.IoTNodes[self.connected_node].x- self.pV[0])/X_MAX, (environment.IoTNodes[self.connected_node].y- self.pV[1] ) /X_MAX]) if self.connected else np.array( [ (self.pD[0] -  self.pV[0])/X_MAX , (self.pD[1] -  self.pV[1])/X_MAX ] )
                               ),
                              axis=0)
        # p_o = np.array([0, 0],  dtype=np.float32)#上一时刻位置
        # p_n = np.array([0, 0], dtype=np.float32)#这一时刻位置
        # v_p = [np.array([0, 0], dtype=np.float32),np.array([0, 0], dtype=np.float32)]#预测的速度值
        # weights=[0.1,0.1,0.2,0.6]
        # for j,pos_q in enumerate(self.jammer_pos_q):
        #     print(environment.JAMMERs[j].vV)
        #     for i,pos in enumerate(pos_q.queue):
        #         if i < 3 : n_s=np.append(n_s,pos[0:2]/X_MAX)
        #         if i == 0:
        #             p_o = copy.deepcopy(pos[0:2])
        #         else:
        #             p_n = copy.deepcopy(pos[0:2])
        #             print("before",v_p[j])
        #             v_p[j] += ((p_n-p_o) / TIMESTAMP * weights[i-1])
        #             print("after",v_p[j])
        #             p_o = copy.deepcopy(pos[0:2])
        # #print(v_p)
        # return n_s
    def in_nfzones(self,environment):
        for nfzone in environment.NFZones:
            if nfzone[0] <= self.pV[0] and  nfzone[1] >= self.pV[0] and nfzone[2] <= self.pV[1] and nfzone[3] >= self.pV[1]:
                return True
        return False
    def action(self,a):#根据动作进行状态转移计算 动作action定义为[x方向加速度,y方向加速度]的np数组
        coll_flag=0#出界计数
        self.pV[0]  =  self.pV[0] + self.vV[0] * TIMESTAMP + 0.5 * a[0] * TIMESTAMP**2
        self.pV[1]  =  self.pV[1] + self.vV[1] * TIMESTAMP + 0.5 * a[1] * TIMESTAMP**2
        self.vV[0]  =  float(np.clip(self.vV[0] + a[0] * TIMESTAMP,-V_MAX,V_MAX))
        self.vV[1]  =  float(np.clip(self.vV[1] + a[1] * TIMESTAMP,-V_MAX,V_MAX))
        # 出界了
        if self.pV[0] <= 0 or self.pV[0]  >= X_MAX:
            coll_flag += 1
        if self.pV[1] <= 0 or self.pV[1]  >= X_MAX:
            coll_flag += 1
        return coll_flag

    def get_snv(self,environment,node):#计算一个节点和自身通信的信噪比
        #先计算所有干扰机在典型机处的干扰
        pj=0
        ns=0.00005#随机噪声
        for jammer in environment.JAMMERs:
            pj += (jammer.pJ * get_p_recv(jammer.pV,self.pV))
        #返回信噪比
        return ( node.p * get_p_recv(np.array([node.x,node.y,0]),self.pV) )/(pj + ns)

    def observe(self,environment,timestep):#观察环境 获得物联网节点、干扰机等其他信息,选择与一个节点进行通信或是断开连接
        self.close_2 = list(map(list(environment.dis_martrix[self.num]).index,heapq.nsmallest(3, environment.dis_martrix[self.num])))[1:]
        self.jammer_rep=np.array([0,0],dtype=np.float32)
        for i,jammer in enumerate(environment.JAMMERs):
            vector=(self.pV-jammer.pV)[0:2]
            cos=vector[0]/(vector[0]**2 + vector[1]**2)**0.5
            sin=vector[1]/(vector[0]**2 + vector[1]**2)**0.5
            self.jammer_rep += (np.array([cos,sin],dtype=np.float32) / (dis(jammer.pV,self.pV,5)))
        #print(self.jammer_rep)
        #返回两个标志 代表：是否完成传输 以及 是否因为信噪比过小而断连
        #part1 获得节点信息
        snv_list=[]#存储节点信噪比的数组
        for node in environment.IoTNodes:
            snv_list.append(self.get_snv(environment=environment,node=node))

        #先是简单的选择信噪比最大的节点完成通信
        snv_list_sorted =sorted(snv_list,reverse = True)#对信噪比序列从大到小排序
        #若是连接状态
        if self.connected:
            #计算上个时间步骤内的数据收集
            data=self.rnv*TIMESTAMP
            #print(self.rnv)
            environment.IoTNodes[self.connected_node].data -= data
            if environment.IoTNodes[self.connected_node].data <= 0:#数据传输完成
                #print("节点",self.connected_node,"传输完成！at:", timestep * TIMESTAMP)
                environment.done_cnt+=1
                self.rnv = 0
                environment.IoTNodes[self.connected_node].done = True
                #print("uav",self.num,"disconnect to node", self.connected_node, "at:", timestep * TIMESTAMP)
                self.disconnect(environment.IoTNodes[self.connected_node])
                #尝试寻找一个新的连接
                for snv in snv_list_sorted:
                    node_num = snv_list.index(snv)
                    if snv < TSV:  # 不满足信噪比要求
                        self.rnv=0
                        return True,False
                    elif not environment.IoTNodes[node_num].connected and not environment.IoTNodes[node_num].done:
                        #print("uav",self.num,"connect to node", node_num, "at:", timestep * TIMESTAMP,"type 0")
                        self.connect(environment.IoTNodes[node_num], node_num)
                        self.rnv = math.log(1 + snv, 2)  # 计算有效信息率
                        return True,False
            # 检查是否有更好的连接节点可选
            for snv in snv_list_sorted:#从大到小遍历
                node_num = snv_list.index(snv)
                if node_num == self.connected_node:#遍历到了当前连接节点 且并未找到更好选择
                    self.rnv = math.log(1 + snv, 2)
                    return False,False
                elif (not environment.IoTNodes[node_num].connected and not environment.IoTNodes[node_num].done) and  (snv > snv_list[self.connected_node] ): #找到了更好的连接节点
                    #print("uav",self.num,"disconnect to node", self.connected_node,"at:",timestep*TIMESTAMP)
                    self.disconnect(environment.IoTNodes[self.connected_node])
                    # print("uav",self.num,"connect to node", node_num, "at:", timestep * TIMESTAMP)
                    self.connect(environment.IoTNodes[node_num], node_num)
                    self.rnv=math.log(1+snv,2)#计算有效信息率
                    return False,False
            if snv_list[self.connected_node] < TSV and self.connected:#当前节点不再满足信噪比要求
                #print("uav",self.num,"disconnect to node", self.connected_node,"at:",timestep*TIMESTAMP)
                self. disconnect(environment.IoTNodes[self.connected_node])
                self.rnv = 0  # 计算有效信息率
                return False,True
        #还未连接 尝试连接
        else:
            for snv in snv_list_sorted:
                node_num=snv_list.index(snv)
                if snv < TSV :#不满足信噪比要求
                    return False,False
                elif not environment.IoTNodes[node_num].connected and not environment.IoTNodes[node_num].done:
                    #print("uav",self.num,"connect to node", node_num, "at:", timestep * TIMESTAMP,"type 1")
                    self.connect(environment.IoTNodes[node_num],node_num)
                    self.rnv = math.log(1 + snv, 2)  # 计算有效信息率
                    return False, False
        return False,False

    def connect(self,iotnode,node_num):#和节点建立连接 传输数据 完成一个时间步内的通信
        self.connected=True
        iotnode.connected=True
        self.connected_node=node_num
        # print(self.num,self.connected_node)

    def disconnect(self,iotnode):#信噪比不满足通信要求，断开连接
        self.connected=False
        iotnode.connected=False
        self.connected_node=-1

    def step(self, environment, timestep, action):
        if self.arrive:
            return 0
        r = 0
        r -= self.action(a=action*A_MAX) * 1000 #执行动作并计算出界惩罚
        if environment.done_cnt < 12:#当传输未完成时 计算跟传输相关部分的奖励
            r += 120 * self.rnv ** 3  # 提高信噪比的奖励
            r += 100 * np.dot(self.vV,[environment.IoTNodes[self.connected_node].x-self.pV[0],environment.IoTNodes[self.connected_node].y-self.pV[1]]) if self.connected else 0
            flag_done,flag_disconnected = self.observe(environment=environment,timestep=timestep)
            # for uav in self.close_2:
            #     r += 50*environment.UAVs[uav].rnv**3#队友无人机提高信噪比的奖励
            r += 2000 * flag_done#完成环境观察并计算完成一个节点的传输奖励
            r -= 2000 * flag_disconnected#因为不满足信噪比要求而断连的惩罚
            r += np.dot(self.jammer_rep,self.vV/V_MAX)*800 #远离干扰机的奖励

        else:#传输完成时 鼓励无人机前往终点
            r += 5000 / (5 + dis(self.pV[0:2], self.pD[0:2], 0)) if dis(self.pV[0:2], self.pD[0:2],
                                                                        0) < 20 else -2000  # 靠近终点的奖励
        s_ = self.normalization_s(environment=environment)

        if dis(self.pV[0:2], self.pD[0:2], 0) <= 1:  # 第一次到达终点
            self.arrive=True
            r += 2000#到达终点的奖励
            r -= 1500 * (len(environment.IoTNodes) - environment.done_cnt)#到达终点时，节点未完成数据传输受到的惩罚
            r = np.clip(r, -R_MAX, R_MAX)
            #print(self.num,"arrive!")
            return  r/R_MAX

        for num in self.close_2:#和其他无人机碰撞的惩罚 d < 0.4 为碰撞发生 （0.4，1）为碰撞缓冲区
            d = environment.dis_martrix[self.num][num]
            if d > 1 :break
            elif  d <= 0.4:
                #print("boom")
                r -= 1200
            elif 0.4 < d  <= 1:
                r -= 480 / d
        r -= 1000 * int(self.in_nfzones(environment=environment))#禁飞区惩罚
        r -= 500 * (dis(np.array([0,0]), X_MAX * s_[-2:], 0)) if self.connected else 100 * (dis(np.array([0,0]), X_MAX * s_[-2:], 0)) #不靠近目标的惩罚 当前连接的节点或是目的地
        r -= 100 #固定惩罚
        r = np.clip(r,-R_MAX,R_MAX)
        return r/R_MAX


class IoTNode:
    def __init__(self,x,y,p,data):
        ##地面坐标
        self.x=x
        self.y=y
        ##传输功率
        self.p=p
        # self.mode='active'
        # self.connect_flag=False
        self.data=data#初步设定为一个int类型，表示剩余需要传输的比特数
        self.connected=False
        self.done=False

def make_env():
    uav_test_0 = TYPICAL(pV=np.array([0, 0, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=0,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    uav_test_1 = TYPICAL(pV=np.array([0, 5, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=1,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    uav_test_2 = TYPICAL(pV=np.array([5, 0, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=2,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    jammer_test_0 = JAMMER(pV=np.array([7, 3, 5], dtype=np.float32),
                           vV=np.array([0,0], dtype=np.float32),
                           r=np.array([2], dtype=np.float32),
                           vM=np.array([3], dtype=np.float32),
                           num=0,
                           pJ=np.array([0.001/3], dtype=np.float32))

    jammer_test_1 = JAMMER(pV=np.array([4, 9, 5], dtype=np.float32),
                           vV=np.array([0,0], dtype=np.float32),
                           r=np.array([2], dtype=np.float32),
                           vM=np.array([3], dtype=np.float32),
                           num=1,
                           pJ=np.array([0.001/3], dtype=np.float32))

    node_test_0 = IoTNode(x=1.8, y=5, p=0.01, data=8.5)
    node_test_1 = IoTNode(x=2.5, y=4.8, p=0.01, data=8.6)
    node_test_2 = IoTNode(x=8.3, y=7, p=0.01, data=5.8)
    node_test_3 = IoTNode(x=2.7, y=4.6, p=0.01, data=8.4)
    node_test_4 = IoTNode(x=5.3, y=7.5, p=0.01, data=7.2)
    node_test_5 = IoTNode(x=8, y=5, p=0.01, data=9.1)
    node_test_6 = IoTNode(x=6.6, y=2.7, p=0.01, data=9.3)
    node_test_7 = IoTNode(x=6, y=3.2, p=0.01, data=8.4)
    node_test_8 = IoTNode(x=7, y=4.2, p=0.01, data=8.4)
    node_test_9 = IoTNode(x=6.6, y=6.7, p=0.01, data=9.3)
    node_test_10 = IoTNode(x=4, y=3.2, p=0.01, data=9.4)
    node_test_11 = IoTNode(x=2.8, y=2.2, p=0.01, data=10.4)

    nf_zone_0 = np.array([5.5, 7.5, 5.5, 7.5])
    nf_zone_1 = np.array([2.5, 3.5, 2.5, 3.5])

    environment_test = ENVIRONMENT(size=np.array([10, 10, 10], dtype=np.float32),
                                   UAVs=[uav_test_0, uav_test_1, uav_test_2],
                                   JAMMERs=[jammer_test_0, jammer_test_1],
                                   IoTNodes=[node_test_0, node_test_1, node_test_2, node_test_3, node_test_4,
                                             node_test_5, node_test_6, node_test_7, node_test_8, node_test_9,
                                             node_test_10, node_test_11],
                                   NFZones=[nf_zone_0,nf_zone_1])

    environment_test.reset()
    return environment_test
def make_env_random(seed=0):
    np.random.seed(seed)
    uav_test_0 = TYPICAL(pV=np.array([0, 0, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=0,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    uav_test_1 = TYPICAL(pV=np.array([0, 5, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=1,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    uav_test_2 = TYPICAL(pV=np.array([5, 0, 10], dtype=np.float32),
                         vV=np.array([0,0], dtype=np.float32),
                         r=np.array([2], dtype=np.float32),
                         vM=np.array([3], dtype=np.float32),
                         num=2,
                         pD=np.array([10, 10, 10], dtype=np.float32)
                         )

    jammer_test_0 = JAMMER(pV=np.array([7, 3, 5], dtype=np.float32),
                           vV=np.array([0,0], dtype=np.float32),
                           r=np.array([2], dtype=np.float32),
                           vM=np.array([3], dtype=np.float32),
                           num=0,
                           pJ=np.array([0.001/3], dtype=np.float32))

    jammer_test_1 = JAMMER(pV=np.array([4, 9, 5], dtype=np.float32),
                           vV=np.array([0,0], dtype=np.float32),
                           r=np.array([2], dtype=np.float32),
                           vM=np.array([3], dtype=np.float32),
                           num=1,
                           pJ=np.array([0.001/3], dtype=np.float32))

    node_test_0 = IoTNode(x=1.8 + np.random.randint(-2, 2), y=5 + np.random.randint(-2, 2), p=0.01, data=8.5)
    node_test_1 = IoTNode(x=2.5 + np.random.randint(-2, 2), y=4.8 + np.random.randint(-2, 2), p=0.01, data=8.6)
    node_test_2 = IoTNode(x=7.3 + np.random.randint(-2, 2), y=6 + np.random.randint(-2, 2), p=0.01, data=8.8)
    node_test_3 = IoTNode(x=2.7 + np.random.randint(-2, 2), y=4.6 + np.random.randint(-2, 2), p=0.01, data=8.4)
    node_test_4 = IoTNode(x=5.3 + np.random.randint(-2, 2), y=7.5 + np.random.randint(-2, 2), p=0.01, data=7.2)
    node_test_5 = IoTNode(x=8 + np.random.randint(-2, 2), y=5 + np.random.randint(-2, 2), p=0.01, data=8.1)
    node_test_6 = IoTNode(x=6.6 + np.random.randint(-2, 2), y=2.7 + np.random.randint(-2, 2), p=0.01, data=8.3)
    node_test_7 = IoTNode(x=6 + np.random.randint(-2, 2), y=3.2 + np.random.randint(-2, 2), p=0.01, data=8.4)
    node_test_8 = IoTNode(x=7 + np.random.randint(-2, 2), y=4.2 + np.random.randint(-2, 2), p=0.01, data=8.4)
    node_test_9 = IoTNode(x=6.6 + np.random.randint(-2, 2), y=6.7 + np.random.randint(-2, 2), p=0.01, data=9.3)
    node_test_10 = IoTNode(x=4 + np.random.randint(-2, 2), y=3.2 + np.random.randint(-2, 2), p=0.01, data=9.4)
    node_test_11 = IoTNode(x=2.8 + np.random.randint(-2, 2), y=2.2 + np.random.randint(-2, 2), p=0.01, data=10.4)

    nf_zone_0 = np.array([5.5, 7.5, 5.5, 7.5])
    nf_zone_1 = np.array([2.5, 3.5, 2.5, 3.5])

    environment_test = ENVIRONMENT(size=np.array([10, 10, 10], dtype=np.float32),
                                   UAVs=[uav_test_0, uav_test_1, uav_test_2],
                                   JAMMERs=[jammer_test_0, jammer_test_1],
                                   IoTNodes=[node_test_0, node_test_1, node_test_2, node_test_3, node_test_4,
                                             node_test_5, node_test_6, node_test_7, node_test_8, node_test_9,
                                             node_test_10, node_test_11],
                                   NFZones=[nf_zone_0,nf_zone_1])

    environment_test.reset()
    return environment_test
