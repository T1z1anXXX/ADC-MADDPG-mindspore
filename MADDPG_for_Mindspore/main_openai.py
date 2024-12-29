from function import *

if __name__ == '__main__':
    arglist = parse_args()
    if arglist.train_model == 'm':
        print("MADDPG")
        train_mix(arglist)
    elif arglist.train_model == 'n':
        print("MADDPG with no jammer")
        train_mix_no_jammer(arglist,type="no_jammer")
    elif arglist.train_model == 'f':
        print("MADDPG with fixed jammer")
        train_mix_fixed_jammer(arglist,type="fixed_jammer")
    elif arglist.train_model == 'a':
        print("CAA-MADDPG")
        train_mix_attention(arglist)
    elif arglist.train_model == 'd':
        print("Double-Q-MADDPG")
        train_mix_double_q(arglist)
    elif arglist.train_model == 'adq':
        print("ADQ-MADDPG")
        train_mix_adq(arglist)


