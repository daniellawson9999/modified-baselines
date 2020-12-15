import numpy as np
import gym
import gym_yumi
import pickle
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', default=False, type=bool, help='whether to evaluate on approach')
    parser.add_argument('--grasp', default=False, type=bool, help='whether to evaluate on grasp')
    parser.add_argument('--approach-policy-path', default=None, type=str, help='path to location of approach policy')
    parser.add_argument('--grasp-policy-path', default=None, type=str, help='path to location of approach policy')
    parser.add_argument('--grasp-location-path', default=None, type=str, help='path to save grasp locations')
    parser.add_argument('--episodes', default=5, type=int, help='number of episodes to test with')
    args = parser.parse_args()
    
    # ex usage: python baselines_test.py --approach=True --approach-policy-path=./models/her9/policy_best.pkl --episodes=5
    # python baselines_test.py --grasp=True --grasp-policy-path=./models/her9/policy_best.pkl  --grasp-location-path=./locations/her10_2.json --episodes=5

    # check for valid arg combinations
    assert (args.grasp or args.approach), "Must specifiy a task"
    if args.approach:
        assert (args.approach_policy_path != None), "must specify approach policy"
    if args.grasp:
        assert (args.grasp_policy_path != None), "must specify grasp policy"
        assert (args.grasp_location_path != None), "must specify grasp locations"


    if args.grasp:
        goal = "grasp"
    else:
        goal = "approach"

    # env_kwargs = {'headless':False, 'maxval': 1, 'random_peg':True, 
    #               'normal_offset':False, 'goals': [goal]} 
    env_kwargs = {'headless':False, 'maxval':1} 

    approach_model = None
    grasp_model = None
    if args.approach:
        approach_model = pickle.load(open(args.approach_policy_path, 'rb'))
    if args.grasp:
        grasp_model = pickle.load(open(args.grasp_policy_path, 'rb'))
        #grasp_model = None


    # single approach test
    if (args.approach and not args.grasp):
        env = gym.make('goal-yumi-pegtransfer-v0', **env_kwargs)
        save_path = args.grasp_location_path
        # if saving for grasp locations enabled, build dictionary
        config_dictionary = {}
        if save_path:
            for peg in env.pegs:
                config_dictionary[peg] = []
        for i in range(args.episodes):
            obs = env.reset()
            for j in range(50):
                actions, _, _, _ = approach_model.step(obs)
                obs, rew, done, info = env.step(actions)
                env.render()
                if rew == 0:
                    if save_path:
                        joint_values = []
                        for k in range(len(env.limb.joints)):
                            joint_values.append(env.limb.joints[k].get_joint_position())
                        config_dictionary[env.peg_name].append(joint_values)
                    break
        if save_path:
            with open(save_path, "w") as save_file:
                save_file.write(json.dumps(config_dictionary))
        env.close()
    # single grasp test
    elif (not args.approach and args.grasp):
        env_kwargs['arm_configs'] = args.grasp_location_path
        #env_kwargs['random_peg_xy'] = True
        #env_kwargs['random_peg'] = False
        env = gym.make('grasp-goal-yumi-pegtransfer-v0', **env_kwargs)
        env.reset()
        for i in range(args.episodes):
                obs = env.reset()
                for j in range(10):
                    actions, _, _, _ = grasp_model.step(obs)
                    actions = actions.tolist()
                    actions[-1] = round(actions[-1])

                    obs, rew, done, info = env.step(actions)
                    env.render()
                    if rew == 0:
                        break
        env.close() 
    # both
    else:
        env_kwargs['arm_configs'] = args.grasp_location_path
        env = gym.make('goal-yumi-pegtransfer-v0', **env_kwargs)
        env.goals = ['approach']
        for i in range(args.episodes):
            obs = env.reset()
            for j in range(50):
                actions, _, _, _ = approach_model.step(obs)
                actions.append(1) # append action for gripper, keep open until grasp task
                obs, rew, done, info = env.step(actions)
                env.render()
                if rew == 0:
                    break
            env.env.goals = ['grasp']
            for j in range(50):
                actions, _, _, _ = grasp_model.step(obs)
                obs, rew, done, info = env.step(actions)
                env.render()
                if rew == 0:
                    break
        env.close()

        

