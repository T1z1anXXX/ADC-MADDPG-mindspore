import torch
import mindspore as ms
import mindspore.nn as nn
import matplotlib.pyplot as plt
import os
from arguments import parse_args
from function import *

'''
for environment
'''
DATA_INIT=[8.5,8.6,8.8,8.4,7.2,8.1,8.3,8.4,8.4,9.3,9.4,10.4]
arglist = parse_args()
obs_shape_n = [15,15,15,8,8]
action_shape_n = [2,2,2,2,2]
'''
for drawing
'''
colors=['green','black','blue','red','orange']
styles = ['bo', 'co', 'yo', 'r*', 'r*']
rw_u = 0
rw_j = 0
col_cnt = 0
nf_cnt = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
dynamic_points = []
text = ax.text(5, -0.5, '', ha='center', va='center')


def pytorch2mindspore(model_path,save_path = r'C:\Users\Lenovo\Desktop\MADDPG_for_Mindspore\ms_model'):# read pth file
    par_dict = torch.load(model_path).state_dict()
    params_list =[]
    for name in par_dict:
        param_dict={}
        parameter =par_dict[name]
        param_dict['name']= name
        param_dict['data']= ms.Tensor(parameter.cpu().numpy())
        params_list.append(param_dict)
        save_dir = save_path + model_path[28:-9]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ms.save_checkpoint(params_list, save_path + model_path[28:-3] + '.ckpt')

def acmodel2mindspore(path):
    models = os.listdir(path)
    for model in models:
        model_path = os.path.join(path, model)
        pytorch2mindspore(model_path)

class abstract_agent_mindspore(nn.Cell):
    def __init__(self):
        super(abstract_agent_mindspore, self).__init__()
    def act(self, input):
        policy, value = self.construct(input)  # flow the input through the nn
        return policy, value

class mindspore_actor(abstract_agent_mindspore):
    def __init__(self, num_inputs, action_size, args):
        super(mindspore_actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Dense(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Dense(args.num_units_1, args.num_units_2)
        self.linear_a3 = nn.Dense(args.num_units_2, args.num_units_2)
        self.linear_a = nn.Dense(args.num_units_2, action_size)

    def construct(self, x, model_original_out=False):
        x = self.LReLU(self.linear_a1(x))
        x = self.LReLU(self.linear_a2(x))
        x = self.LReLU(self.linear_a3(x))
        model_out = self.tanh(self.linear_a(x))
        u = ms.ops.rand_like(model_out)
        # print(model_out)
        policy = ms.ops.clip(model_out - ms.ops.log(-ms.ops.log(u)), -1, 1)
        # policy = 2 * policy - 1 #从(0,1)转换到(-1,1)
        if model_original_out == True:   return model_out, policy  # for model_out criterion
        return policy

def get_mix_trainers_mindspore(model_name=r'C:\Users\Lenovo\Desktop\MADDPG_for_Mindspore\ms_model\double_q_2408_221811_2202000'):
    """ load the model """
    actors_param_dict = [ms.load_checkpoint(model_name+r'\a_c_{}.ckpt'.format(agent_idx)) \
        for agent_idx in range(5)]
    actors=[]
    for i,param_dict in enumerate(actors_param_dict):
        actor = mindspore_actor(obs_shape_n[i], action_shape_n[i], arglist)
        ms.load_param_into_net(actor, param_dict)
        actors.append(actor)
    return actors

def fig_init(env):
    global fig,ax,dynamic_points,text
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-2, 12)  # 设置 x 轴范围为 0 到 10
    ax.set_ylim(-2, 12)  # 设置 y 轴范围为 -1 到
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(np.arange(-2, 12, 1))
    plt.yticks(np.arange(-2, 12, 1))
    text = ax.text(5, -0.5, '', ha='center', va='center')
    ax.set_aspect(1)
    rect = plt.Rectangle((env.NFZones[0][0], env.NFZones[0][2]),
                         env.NFZones[0][1] - env.NFZones[0][0],
                         env.NFZones[0][3] - env.NFZones[0][2],
                         facecolor='blue', edgecolor='red', alpha=0.5)
    ax.add_patch(rect)
    rect = plt.Rectangle((env.NFZones[1][0], env.NFZones[1][2]),
                         env.NFZones[1][1] - env.NFZones[1][0],
                         env.NFZones[1][3] - env.NFZones[1][2],
                         facecolor='blue', edgecolor='red', alpha=0.5)
    ax.add_patch(rect)
    start_0 = plt.Circle(xy=(0, 0), radius=0.4, color=colors[0], fill=False)
    start_1 = plt.Circle(xy=(0, 5), radius=0.4, color=colors[1], fill=False)
    start_2 = plt.Circle(xy=(5, 0), radius=0.4, color=colors[2], fill=False)
    dest = plt.Circle(xy=(10, 10), radius=1, color='yellow', fill=False)
    ax.add_patch(start_0)
    ax.add_patch(start_1)
    ax.add_patch(start_2)
    ax.add_patch(dest)
    d_x=[]
    d_y=[]
    for uav in chain(env.UAVs,env.JAMMERs):
        d_x.append(uav.pV[0])
        d_y.append(uav.pV[1])
    dynamic_points = [ax.plot([], [], style)[0] for style in styles]

def test_once():

    env = make_env()
    env_random = make_env_random()
    obs_n_random = env_random.reset()
    arglist = parse_args()
    actors = get_mix_trainers_mindspore()
    obs_n = env.reset()
    episode_step = 0
    SR_n, DR_n, CR_n, IR_n, rw_j_n, rw_u_n = [], [], [], [], [], []
    fig_init(env = env_random)

    rw_u = 0
    rw_j = 0
    nf_cnt = 0
    col_cnt = 0

    s_x = []
    s_y = []
    for i, node in enumerate(env.IoTNodes):
        s_x.append(node.x)
        s_y.append(node.y)
        if i == 0:
            ax.plot(s_x, s_y, 'k^', label='Nodes')
        else:
            ax.plot(s_x, s_y, 'k^')
    plt.legend()
    for step in range(400):
        action_n = []
        for actor, obs in zip(actors, obs_n):
            model_out, _ = actor(ms.tensor(obs), model_original_out=True)
            action_n.append(model_out.asnumpy())
        obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_step, jammer_act=arglist.jammer_act)
        for i, uav in enumerate(env.UAVs):
            if step == 0:
                ax.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num], label='UAV_' + str(i), markersize='4')
            else:
                ax.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num], markersize='4')
            # if uav.num==1:print(uav.vV)
        for j, jammer in enumerate(env.JAMMERs):
            if step == 0:
                ax.plot(jammer.pV[0], jammer.pV[1], '+', color=colors[jammer.num + 3], label='Jammer_' + str(j),
                        markersize='4')
            else:
                ax.plot(jammer.pV[0], jammer.pV[1], '+', color=colors[jammer.num + 3], markersize='4')
        for node in env.IoTNodes:
            if node.done:
                ax.plot(node.x, node.y, 'g^')
        rw_u += np.sum(rew_n[:3])
        rw_j += np.sum(rew_n[-2:])
        for uav in env.UAVs:
            if uav.in_nfzones(env):
                nf_cnt += 1
        for i in range(2):
            for j in range(i + 1, 2):
                if env.dis_martrix[i][j] <= 0.4:
                    col_cnt += 1
        if all(done_n[0:3]):
            break
    plt.legend()
    plt.savefig('static_pictures/' + 'double_q_2408_221811_2202000.jpg', dpi=600,
                bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # acmodel2mindspore(r'D:\study_bz\大四\models_backup\double_q_2408_221811_2202000')
    test_once()

