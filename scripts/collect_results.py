import os

def read_results():
    folder_name = "data/models/garat/" + "Double_Hopper_10K_GAIL_sim2real_TRPO_2000000_10000_50_"
    # Double_SAC_reset_Hopper_GAIL_sim2real_SAC_2000000_3000_50_
    # Double_SAC_reset_Walker_GAIL_sim2real_SAC_2000000_3000_50_
    # Single_SAC_reset_Hopper_GAIL_sim2real_SAC_2000000_3000_50_
    # Single_SAC_reset_Walker_GAIL_sim2real_SAC_2000000_3000_50_
    total_runs = 5
    total_groundings = 1
    deter_eval = []
    stochastic_eval = []
    for i in range(total_runs):
        base_path = folder_name + str(i+1) + "/grounding_step_"

        for g in range(total_groundings):
            deterministic_eval_path = base_path + str(g) +"/output.txt"
            stochastic_eval_path = base_path + str(g) + "/stochastic_output.txt"

            with open(deterministic_eval_path, 'r') as f:
                deter_eval.append(f.read())
            with open(stochastic_eval_path, 'r') as f:
                stochastic_eval.append(f.read())


    with open(folder_name + "eval", 'w') as f:
        for g in range(total_groundings):
            f.write("Grounding step:" + str(g) +"\n")
            for i in range(total_runs):
                # print(deter_eval[g*total_groundings + i])
                f.write(deter_eval[i*total_groundings + g])
    with open(folder_name + "stochastic_eval", 'w') as f:
        for g in range(total_groundings):
            f.write("Grounding step:" + str(g) +"\n")
            for i in range(total_runs):
                f.write(stochastic_eval[i*total_groundings + g])

if __name__ == '__main__':
    read_results()
    os._exit(0)