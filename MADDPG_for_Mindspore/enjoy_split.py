
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from function import *
colors=['green','black','blue','red','orange']
styles = ['bo', 'co', 'yo', 'r*', 'r*']
DATA_INIT=[8.5,8.6,8.8,8.4,7.2,8.1,8.3,8.4,8.4,9.3,9.4,10.4]




env = make_env()
env_random = make_env_random()
obs_n_random = env_random.reset()
arglist = parse_args()
actors = get_mix_trainers(arglist)
obs_n = env.reset()
episode_step = 0

rw_u = 0
rw_j = 0
col_cnt = 0
nf_cnt = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
dynamic_points = []
text = ax.text(5, -0.5, '', ha='center', va='center')
def fig_init():
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

def init():
    rect = plt.Rectangle((env.NFZones[0][0], env.NFZones[0][2]),
                         env.NFZones[0][1] - env.NFZones[0][0],
                         env.NFZones[0][3] - env.NFZones[0][2],
                         fill=False)
    ax.add_patch(rect)
    return dynamic_points,
def update(frame):
    global obs_n,rw_u,rw_j,env,col_cnt,nf_cnt,dynamic_points,text
    action_n = []
    for actor, obs in zip(actors, obs_n):
        model_out, _ = actor(torch.tensor(obs).to(arglist.device, torch.float), model_original_out=True)
        action_n.append(model_out.detach().cpu().numpy())
    obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_step,jammer_act=arglist.jammer_act)
    rw_u += np.sum(rew_n[:3])
    rw_j += np.sum(rew_n[-2:])
    if all(done_n[0:3]):
        return dynamic_points
    for i in range(2):
        for j in range(i + 1, 2):
            if env.dis_martrix[i][j] <= 0.4:
                col_cnt+=1
    for uav in env.UAVs:
        if uav.in_nfzones(env):
            nf_cnt += 1

    rw_u += np.sum(rew_n[:3])
    rw_j += np.sum(rew_n[-2:])
    #text.set_text('rw_u:{:.3f} rw_j:{:.3f}'.format(rw_u, rw_j))
    for i,uav in enumerate(chain(env.UAVs,env.JAMMERs)):
        dynamic_points[i].set_data(uav.pV[0], uav.pV[1])
    for node in env.IoTNodes:
        if node.done:
            ax.plot(node.x,node.y,'g^')
    return dynamic_points

def enjoy_mix(arglist, random = False, save = True, test_rounds = 100, model_name='',dynamic=True):
    global env,env_random,obs_n,rw_u,rw_j,col_cnt,nf_cnt,fig, ax
    SR_n,DR_n,CR_n,IR_n,rw_j_n,rw_u_n=[],[],[],[],[],[]
    for round in range(test_rounds):
        if save:
            fig_init()
        rw_u = 0
        rw_j = 0
        col_cnt = 0
        nf_cnt = 0
        env = make_env_random(seed=round) if random else make_env()
        obs_n = env.reset(random = False)

        if save and dynamic:
            s_x = []
            s_y = []
            for node in env.IoTNodes:
                s_x.append(node.x)
                s_y.append(node.y)
            static_points, = ax.plot(s_x, s_y, 'k^', animated=True)

            ani=FuncAnimation(fig, update,  frames=np.linspace(start = 0, stop = 20, num = 400)
        , interval=50, blit=True)
            ani.save(arglist.mix_model_name[7:-2]+ ".gif", fps=30, writer="imagemagick",dpi=300)

        elif save and not dynamic:
            s_x = []
            s_y = []
            for i,node in enumerate(env.IoTNodes):
                s_x.append(node.x)
                s_y.append(node.y)
                if i == 0:
                    ax.plot(s_x, s_y, 'k^',label = 'Nodes')
                else:
                    ax.plot(s_x, s_y, 'k^')
            plt.legend()
            for step in range(400):
                action_n = []
                for actor, obs in zip(actors, obs_n):
                    model_out, _ = actor(torch.tensor(obs).to(arglist.device, torch.float), model_original_out=True)
                    action_n.append(model_out.detach().cpu().numpy())
                obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_step, jammer_act=arglist.jammer_act)
                for i,uav in enumerate(env.UAVs):
                    if step == 0:
                        ax.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num], label = 'UAV_'+str(i), markersize='4')
                    else:
                        ax.plot(uav.pV[0], uav.pV[1], '.', color=colors[uav.num], markersize='4')
                    # if uav.num==1:print(uav.vV)
                for j,jammer in enumerate(env.JAMMERs):
                    if step == 0:
                        ax.plot(jammer.pV[0], jammer.pV[1], '+', color=colors[jammer.num+3], label = 'Jammer_'+str(j), markersize='4')
                    else:
                        ax.plot(jammer.pV[0], jammer.pV[1], '+', color=colors[jammer.num+3], markersize='4')
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
                if model_name == '':
                    model_name = arglist.mix_model_name[7:-2]
                # if step % 20 == 0:
                #     plt.legend()
                #     plt.savefig('static_pictures/' + model_name + '_' + str(step) + 'static.jpg', dpi=600,
                #                 bbox_inches='tight')
            plt.legend()
            plt.savefig('static_pictures/' + model_name + '_' +  'static.jpg', dpi=600,
                        bbox_inches='tight')
            plt.close()

        elif not save:
            for step in range(400):
                action_n = []
                for actor, obs in zip(actors, obs_n):
                    model_out, _ = actor(torch.tensor(obs).to(arglist.device, torch.float), model_original_out=True)
                    action_n.append(model_out.detach().cpu().numpy())
                obs_n, rew_n, done_n = env.step(action_n=action_n, timestep=episode_step,jammer_act=arglist.jammer_act)
                #for uav in env.UAVs:
                    #print(uav.vV)
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


        data_all = sum(DATA_INIT)
        data_left = 0
        for node in env.IoTNodes:
            data_left += node.data if node.data > 0 else 0
        SR_n.append(env.done_cnt / 12)
        DR_n.append((data_all-data_left)/data_all)
        CR_n.append(col_cnt / 400)
        IR_n.append(nf_cnt / 1200)
        rw_j_n.append(rw_j)
        rw_u_n.append(rw_u)

        # print("SR:", env.done_cnt / 12)
        # print("DR:",(data_all-data_left)/data_all)
        # print("CR:", col_cnt / 400)
        # print("IR:", nf_cnt / 1200)
        # print('rw_u:{} rw_j:{}'.format(rw_u, rw_j))

    if model_name == '':
        print("model name:",arglist.mix_model_name[7:-2])
    else:
        print("model name:", model_name)
    print("mean SR:", np.mean(SR_n))
    print("mean DR:", np.mean(DR_n))
    print("mean CR:", np.mean(CR_n))
    print("mean IR:", np.mean(IR_n))


def draw_rw(filenames,draw_epidodes,labels,draw_jammer=False,colors=['black','red','blue','green','orange','yellow'],save=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, (filename,label,color) in enumerate(zip(filenames,labels,colors)):
        data=pd.read_csv(filename)
        mean_rw_u=data['mean_rw_t'][0:draw_epidodes]
        mean_rw_j=data['mean_rw_j'][0:draw_epidodes]
        var_rw_u = data['var_rw_t'][0:draw_epidodes]
        var_rw_j = data['mean_rw_j'][0:draw_epidodes]
        episode=np.array(range(1,draw_epidodes+1))
        ax.plot(episode,mean_rw_u,color=color,label=label)
        plt.fill_between(episode, mean_rw_u - np.sqrt(var_rw_u), mean_rw_u + np.sqrt(var_rw_u), color=color, alpha=0.4)
        if draw_jammer:
            ax.plot(episode,mean_rw_j,color=colors[i+3],label=label + '_jammer')
            plt.fill_between(episode, mean_rw_j - np.sqrt(var_rw_j), mean_rw_j + np.sqrt(var_rw_j), color='violet',
                             alpha=0.2)
        plt.legend()
    ax.set_title('Reward-Episode Plots of MADDPG and its enhanced Algorithms for UAVs')
    ax.set_xlabel('episodes')
    ax.set_ylabel('rewards')
    if save:
        plt.savefig('rewards.jpg',dpi=600)
    plt.show()
    
if __name__ == '__main__':
    enjoy_mix(arglist, random = False, save = True, test_rounds = 1, dynamic = False)
    # draw_rw(filenames=['csv/rw_mix_v2.csv','csv/rw_attention_v3.csv','csv/rw_mix_double_q_v3.csv','csv/rw_adq.csv'],draw_epidodes=600,\
    #          labels=['MADDPG','AC-MADDPG','DC-MADDPG', 'ADC-MADDPG'],draw_jammer=False,save=True)
    #enjoy_typical(arglist)
    #enjoy_jammer(arglist)
    #model_list=['attention_2404_291102_7602000','attention_2404_290930_6402000','attention_2404_290915_6202000','attention_2404_260125_6002000','attention_2404_260114_5802000',
                #'attention_2404_080809_6002000','mix_2404_291001_6802000','mix_2404_290946_6602000','mix_2404_290931_6402000','mix_2404_290916_6202000',
                #'mix_2404_010746_6002000','mix_2404_010731_5802000']
    # model_list = ['attention_2404_290930_6402000','attention_2404_080809_6002000',
    #               'double_q_2405_042311_6402000','double_q_2405_042300_6202000',
    #               'adq_2405_052026_6602000','adq_2405_052014_6402000',
    #               'adq_2405_052038_6802000','adq_2405_052002_6202000',
    #               'mix_2404_291001_6802000','mix_2404_290946_6602000']

    # model_list = ['mix_2404_291001_6802000','attention_2404_080809_6002000',
    #               'double_q_2405_042311_6402000','adq_2405_052002_6202000']
    # model_list = ['mix_2404_290916_6202000','mix_2404_010746_6002000','mix_2404_010731_5802000','mix_2404_010717_5602000']
    # for model_name in model_list:
    #     actors = get_mix_trainers(arglist,model_name)
    #     enjoy_mix(arglist,random=True,save=False,test_rounds=100,model_name=model_name)

    # env.reset(jammer=False)
    # enjoy_mix(arglist, random=False, save=True, test_rounds=1,dynamic=True)
    # env.reset(jammer=False)
    #enjoy_mix(arglist, random=False, save=True, test_rounds=1, dynamic=False)

    #enjoy_mix(arglist, random=True, save=True, test_rounds=1, dynamic=True)

    # draw_rw(filenames=['rw_mix_fixed_jammer.csv'], draw_epidodes=800,
    #         labels=['MADDPG with fixed jammer'],save=True)
    #plt.savefig('no_jammer.png')
