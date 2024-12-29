
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a3 = nn.Linear(args.num_units_2, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain_tanh)
        self.linear_a3.weight.data.mul_(gain_tanh)
        self.linear_a.weight.data.mul_(gain_tanh)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.tanh(self.linear_a2(x))
        x = self.tanh(self.linear_a3(x))
        policy = self.tanh(self.linear_a(x))
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1*2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class openai_critic_double_q(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic_double_q, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c3 = nn.Linear(args.num_units_openai, 1)

        self.linear_c4 = nn.Linear(action_shape_n + obs_shape_n, args.num_units_openai)
        self.linear_c5 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c6 = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c6.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = torch.cat([obs_input, action_input], dim=1)

        x1 = self.LReLU(self.linear_c1(x_cat))
        x1 = self.LReLU(self.linear_c2(x1))
        x1 = self.linear_c3(x1)

        x2 = self.LReLU(self.linear_c4(x_cat))
        x2 = self.LReLU(self.linear_c5(x2))
        x2 = self.linear_c6(x2)
        return x1,x2

    def q1(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = torch.cat([obs_input, action_input], dim=1)

        x1 = self.LReLU(self.linear_c1(x_cat))
        x1 = self.LReLU(self.linear_c2(x1))
        x1 = self.linear_c3(x1)

        return x1

class critic_attention_v2(abstract_agent):
    def __init__(self,obs_dim,act_dim, args):
        super(critic_attention_v2, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.per_obs_dim = args.per_obs_dim
        self.per_act_dim = args.per_act_dim
        #for attention
        self.single_mlp = nn.Linear(self.per_act_dim + self.per_obs_dim, int(args.num_units_openai/4))#17 -> 64
        self.linear_cq = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ck = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_cv = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32

        self.linear_ca = nn.Linear(int(args.num_units_openai/4), 1)  #64->1 #用单层MLP代替矩阵来计算相似度 让q_self k_other通过
        #
        self.linear_c1 = nn.Linear(int(args.num_units_openai/4), args.num_units_openai) #64->256
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c  = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_cq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ck.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_cv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ca.weight, gain=0.1)

        nn.init.xavier_uniform_(self.single_mlp.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, act_input):
        """
        input_g: input_global, input features of all agents
        """
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        v_others = torch.tensor([])
        attentions = torch.tensor([])
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            if i == 0:
                v_others = v_other.unsqueeze(0)
                attentions = a_other.unsqueeze(0)
            else:
                v_others = torch.cat((v_others,v_other.unsqueeze(0)),dim = 0)
                attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        #对a进行softmax
        attentions = attentions/(float(256))**0.5
        attentions = F.softmax(attentions, dim=0)
        #加权求和
        v_others = attentions * v_others
        v_others = torch.sum(v_others,dim = 0)
        x = torch.cat((v_self,v_others),dim = 1)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x = self.LReLU(self.linear_c1(x))
        x = self.LReLU(self.linear_c2(x))
        value = self.linear_c(x)
        return value

class critic_attention_v3(abstract_agent):
    def __init__(self,obs_dim,act_dim, args):
        super(critic_attention_v3, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.per_obs_dim = args.per_obs_dim
        self.per_act_dim = args.per_act_dim
        #for attention
        self.single_mlp = nn.Linear(self.per_act_dim + self.per_obs_dim, int(args.num_units_openai/4))#17 -> 64
        self.linear_cq = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ck = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_cv = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32

        self.linear_ca = nn.Linear(int(args.num_units_openai/4), 1)  #64->1 #用单层MLP代替矩阵来计算相似度 让q_self k_other通过
        self.linear_c1 = nn.Linear(int(args.num_units_openai/8), args.num_units_openai) #32->256
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c  = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_cq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ck.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_cv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ca.weight, gain=0.1)

        nn.init.xavier_uniform_(self.single_mlp.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, act_input):
        """
        input_g: input_global, input features of all agents
        """
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        k_self = self.linear_ck(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        #自注意力
        a_self = self.linear_ca(torch.cat((q_self, k_self), dim=1))
        v_all = v_self.unsqueeze(0)
        attentions = a_self.unsqueeze(0)
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            v_all = torch.cat((v_all,v_other.unsqueeze(0)),dim = 0)
            attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        # 对a进行softmax
        attentions = attentions/(float(32))**0.5
        attentions = F.softmax(attentions, dim=0)
        # 加权求和
        v_all = attentions * v_all
        v_all = torch.sum(v_all,dim = 0)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x = self.LReLU(self.linear_c1(v_all))
        x = self.LReLU(self.linear_c2(x))
        value = self.linear_c(x)
        return value
class critic_attention_new(abstract_agent):
    def __init__(self,obs_dim,act_dim, args):
        super(critic_attention_new, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.per_obs_dim = args.per_obs_dim
        self.per_act_dim = args.per_act_dim
        self.linear_c0 = nn.Linear(self.per_act_dim + self.per_obs_dim, int(args.num_units_openai/2))
        self.linear_c1 = nn.Linear(args.num_units_openai,1)  #求取注意力权重
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c3 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)


        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c0.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight , gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, act_input):
        """
        input_g: input_global, input features of all agents
        """
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        for i,(obs_other,act_other) in enumerate(zip(obs_others,act_others)):
            if i == 0:
                x_cat_others=torch.cat((obs_other,act_other),dim=1)
                x_cat_others = torch.unsqueeze(x_cat_others, dim = 0)
            else:
                x_cat_others=torch.cat((x_cat_others,torch.unsqueeze(torch.cat((obs_other,act_other),dim=1),dim=0)),dim=0)


        #经过单层MLP作用
        x_cat_self = self.LReLU(self.linear_c0(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            if i == 0:
                x_others = torch.unsqueeze(self.tanh(self.linear_c0(x_cat_other)),dim=0)
            else:
                x_others = torch.cat((x_others,torch.unsqueeze(self.tanh(self.linear_c0(x_cat_other)),dim=0)),dim=0)
        #求取注意力权重
        #attentions = [self.LReLU(self.linear_c1(torch.cat([x_cat_self,x_other],dim = 1))) for x_other in x_others]
        for i,x_other in enumerate(x_others):
            if i == 0:
                attentions = torch.unsqueeze(self.LReLU(self.linear_c1(torch.cat([x_cat_self,x_other],dim = 1))),dim = 0)
            else:
                attentions = torch.cat((attentions,torch.unsqueeze(self.LReLU(self.linear_c1(torch.cat([x_cat_self,x_other],dim = 1))),dim = 0)))
        #softmax
        attentions = F.softmax(attentions,dim = 0)
        #加权求和
        for i,(attention,x_other) in enumerate(zip(attentions,x_others)):
            if i == 0:
                x_ = torch.unsqueeze(x_other*attention,dim=0)
            else:
                x_ = torch.cat((x_,torch.unsqueeze(x_other*attention,dim=0)),dim = 0)
        x_sum_others = torch.sum(x_,dim=0)
        x = self.LReLU(self.linear_c2(torch.cat([x_cat_self,x_sum_others], dim=1)))
        x = self.LReLU(self.linear_c3(x))
        value = self.linear_c(x)
        return value

class critic_adq(abstract_agent):
    def __init__(self,obs_dim,act_dim, args):
        super(critic_adq, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.per_obs_dim = args.per_obs_dim
        self.per_act_dim = args.per_act_dim
        #for attention
        self.single_mlp = nn.Linear(self.per_act_dim + self.per_obs_dim, int(args.num_units_openai/4))#17 -> 64
        self.linear_cq = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ck = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_cv = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ca = nn.Linear(int(args.num_units_openai/4), 1)  #64->1 #用单层MLP代替矩阵来计算相似度 让q_self k_other通过
        #
        self.linear_c01 = nn.Linear(int(args.num_units_openai/4), args.num_units_openai) #64->256
        self.linear_c02 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c0  = nn.Linear(args.num_units_openai, 1)


        self.linear_c11 = nn.Linear(int(args.num_units_openai/4), args.num_units_openai) #64->256
        self.linear_c12 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c1  = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_cq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ck.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_cv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ca.weight, gain=0.1)

        nn.init.xavier_uniform_(self.single_mlp.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c01.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c02.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c0.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.linear_c11.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c12.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
    def forward(self, obs_input, act_input):
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        v_others = torch.tensor([])
        attentions = torch.tensor([])
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            if i == 0:
                v_others = v_other.unsqueeze(0)
                attentions = a_other.unsqueeze(0)
            else:
                v_others = torch.cat((v_others,v_other.unsqueeze(0)),dim = 0)
                attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        #对a进行softmax
        attentions = attentions/(float(256))**0.5
        attentions = F.softmax(attentions, dim=0)
        #加权求和
        v_others = attentions * v_others
        v_others = torch.sum(v_others,dim = 0)
        x = torch.cat((v_self,v_others),dim = 1)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x1 = self.LReLU(self.linear_c01(x))
        x2 = self.LReLU(self.linear_c11(x))
        x1 = self.LReLU(self.linear_c02(x1))
        x2 = self.LReLU(self.linear_c12(x2))
        x1 = self.linear_c0(x1)
        x2 = self.linear_c1(x2)
        return x1,x2

    def q1(self,obs_input, act_input):
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        v_others = torch.tensor([])
        attentions = torch.tensor([])
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            if i == 0:
                v_others = v_other.unsqueeze(0)
                attentions = a_other.unsqueeze(0)
            else:
                v_others = torch.cat((v_others,v_other.unsqueeze(0)),dim = 0)
                attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        #对a进行softmax
        attentions = attentions/(float(256))**0.5
        attentions = F.softmax(attentions, dim=0)
        #加权求和
        v_others = attentions * v_others
        v_others = torch.sum(v_others,dim = 0)
        x = torch.cat((v_self,v_others),dim = 1)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x1 = self.LReLU(self.linear_c01(x))

        x1 = self.LReLU(self.linear_c02(x1))

        x1 = self.linear_c0(x1)

        return x1

class critic_adq_v2(abstract_agent):
    def __init__(self,obs_dim,act_dim, args):
        super(critic_adq_v2, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.per_obs_dim = args.per_obs_dim
        self.per_act_dim = args.per_act_dim
        #for attention
        self.single_mlp = nn.Linear(self.per_act_dim + self.per_obs_dim, int(args.num_units_openai/4))#17 -> 64
        self.linear_cq = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ck = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_cv = nn.Linear(int(args.num_units_openai/4), int(args.num_units_openai/8))  # 64->32
        self.linear_ca = nn.Linear(int(args.num_units_openai/4), 1)  #64->1 #用单层MLP代替矩阵来计算相似度 让q_self k_other通过

        self.linear_c11 = nn.Linear(int(args.num_units_openai/8), args.num_units_openai) #32->256
        self.linear_c12 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c1  = nn.Linear(args.num_units_openai, 1)

        self.linear_c21 = nn.Linear(int(args.num_units_openai/8), args.num_units_openai) #32->256
        self.linear_c22 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c2  = nn.Linear(args.num_units_openai, 1)


        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_cq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ck.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_cv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear_ca.weight, gain=0.1)
        nn.init.xavier_uniform_(self.single_mlp.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.linear_c11.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c12.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.linear_c21.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c22.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, act_input):
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        k_self = self.linear_ck(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        #自注意力
        a_self = self.linear_ca(torch.cat((q_self, k_self), dim=1))
        v_all = v_self.unsqueeze(0)
        attentions = a_self.unsqueeze(0)
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            v_all = torch.cat((v_all,v_other.unsqueeze(0)),dim = 0)
            attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        #对a进行softmax
        attentions = attentions/(float(32))**0.5
        attentions = F.softmax(attentions, dim=0)
        #加权求和
        v_all = attentions * v_all
        v_all = torch.sum(v_all,dim = 0)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x1 = self.LReLU(self.linear_c11(v_all))
        x2 = self.LReLU(self.linear_c21(v_all))
        x1 = self.LReLU(self.linear_c12(x1))
        x2 = self.LReLU(self.linear_c22(x2))
        x1 = self.linear_c1(x1)
        x2 = self.linear_c2(x2)
        return x1, x2

    def q1(self,obs_input, act_input):
        #分离当前智能体和其它智能体的信息
        obs_self=obs_input[:,:self.per_obs_dim]
        obs_others=torch.chunk(obs_input[:,self.per_obs_dim:],chunks=int(obs_input.size()[1]/self.per_obs_dim)-1, dim=1)
        act_self=act_input[:,:self.per_act_dim]
        act_others=torch.chunk(act_input[:,self.per_act_dim:],chunks=int(act_input.size()[1]/self.per_act_dim)-1, dim=1)
        x_cat_self=torch.cat((obs_self, act_self), dim=1)
        x_cat_others=[]
        for (obs_other,act_other) in zip(obs_others,act_others):
            x_cat_other = torch.cat((obs_other,act_other),dim=1)
            x_cat_others.append(x_cat_other)
        #先通过单层mlp初步提取特征
        x_cat_self = self.LReLU(self.single_mlp(x_cat_self))
        for i,x_cat_other in enumerate(x_cat_others):
            x_cat_others[i] = self.LReLU(self.single_mlp(x_cat_other))
        #计算Q、K、V
        q_self = self.linear_cq(x_cat_self)
        k_self = self.linear_ck(x_cat_self)
        v_self = self.linear_cv(x_cat_self)
        v_others = torch.tensor([])
        attentions = torch.tensor([])
        #自注意力
        a_self = self.linear_ca(torch.cat((q_self, k_self), dim=1))
        v_all = v_self.unsqueeze(0)
        attentions = a_self.unsqueeze(0)
        #为每个队友计算交叉注意力
        for i,x_cat_other in enumerate(x_cat_others):
            k_other = self.linear_ck(x_cat_other)
            v_other = self.linear_cv(x_cat_other)
            a_other = self.linear_ca(torch.cat((q_self,k_other),dim = 1))
            v_all = torch.cat((v_all,v_other.unsqueeze(0)),dim = 0)
            attentions = torch.cat((attentions, a_other.unsqueeze(0)), dim = 0)
        #对a进行softmax
        attentions = attentions/(float(256))**0.5
        attentions = F.softmax(attentions, dim=0)
        #加权求和
        v_all = attentions * v_all
        v_all = torch.sum(v_all,dim = 0)
        # x = self.LReLU(v_all.permute(1,0,2).reshape(1024,-1))#v_all.reshape(1024,-1)结果不对
        # 进入后续神经网络
        x = self.LReLU(self.linear_c11(v_all))
        x = self.LReLU(self.linear_c12(x))
        value = self.linear_c1(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a3 = nn.Linear(args.num_units_2, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a3.weight, gain=gain)
        nn.init.xavier_uniform_(self.linear_a.weight, gain=gain_tanh)
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        x = self.LReLU(self.linear_a3(x))
        model_out = self.tanh(self.linear_a(x))
        #model_out = 2 * model_out - 1
        u = torch.rand_like(model_out)
        #print(model_out)
        policy = torch.clip(model_out - torch.log(-torch.log(u)),-1,1)
        #policy = 2 * policy - 1 #从(0,1)转换到(-1,1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy
