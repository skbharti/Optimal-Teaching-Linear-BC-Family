import os, sys, shutil, time, pickle
import pandas as pd
import yaml
from joblib import Parallel, delayed
import numpy as np

from instance_generator import generate_polygon_tower, generate_pick_the_diamond, generate_grid_world_navigation
from teaching import find_optimal_teaching_set, find_random_teaching_set
from consistent_learners import train_classifier, evaluate_classifier
from helper import process_text, print_state

def set_output_directories_and_path(args):

    if(args['record']):

        '''
            - if recording is on, read summary file and assign a unique run_directory to this run
                - create run_directory and a new entry for this run in runs_summary.csv file
                - save the code files
            - else set the output directory to 'no_record
        '''

        # read summary file as csv
        run_summary_path = os.path.join(args['project_base_path'], args['output_dir'], 'runs_summary.csv')
        summary_df = pd.read_csv(run_summary_path)
        summary_df = summary_df.loc[:, ~summary_df.columns.str.contains('^Unnamed')]

        # fetch the current run_id, add a run entry and save the run_summary csv file
        run_id = summary_df.iloc[[-1]]['run_id']
        current_run_id = int(run_id) + 1

        args['current_run_id'] = current_run_id
        args['run_output_dir'] = args['project_tag']+'-'+str(current_run_id)
        summary_df = summary_df.append({'run_id':current_run_id, 'run_dir':args['run_output_dir'], 'params':args,'notes':args['run_notes'], 'others':'Nothing'}, ignore_index=True)
        summary_df.to_csv(run_summary_path)

        # populate the path to save the run_output and run_code
        args['run_output_path'] = os.path.join(args['project_base_path'], args['output_dir'], args['run_output_dir'])
        args['run_code_path'] = os.path.join(args['run_output_path'], 'code')
        print('Recording On, saving data in ', args['run_output_path'])

    else:
        # by default save the outputs to 'no_record' directory
        args['run_output_path'] = os.path.join(args['project_base_path'], args['output_dir'], 'no_record')
        args['run_code_path'] = os.path.join(args['run_output_path'], 'code')
        print('Recording Off, saving data in ', args['run_output_path'])

    # create the unique run directory
    if not os.path.exists(args['run_code_path']):
        os.makedirs(args['run_code_path'])
    
    # copy the codes to be saved to run_output_dir
    for file in args['save_file_list']:
        file_name = os.path.basename(file)  # 'file' is relative path wrt project_base_dir
        shutil.copyfile(os.path.join(args['project_base_path'], file), os.path.join(args['run_code_path'], file_name))



def create_run_summary_file(project_tag : str, output_base_dir  : str):
    
    summary_df = pd.DataFrame(columns=['run_id','run_dir','params','run_note','extra_info'])
    summary_df = summary_df.append({'run_id':0, 'run_dir':project_tag+'_0', 'params':'no_params', \
                                    'run_note':'starting example entry', 'extra_info':'nothing'}, ignore_index=True)

    files_list = os.listdir(output_base_dir)
    if('run_summary.csv' in files_list):
        raise ValueError('Summary file already present!')
    else:
        summary_df.to_csv(os.path.join(output_base_dir, 'run_summary.csv'))


def initialize_run_summary():
    
    output_base_dir = 'outputs'

    if(not os.path.exists(output_base_dir)):
        os.makedirs(output_base_dir)

    create_run_summary_file(project_tag='TEACH', output_base_dir=output_base_dir)


def run_code(args):
    
    if(args['terminal_output_to_log_file'] and not args['run_parallel']):
        old_stdout = sys.stdout
        log_file = open(os.path.join(args['run_output_path'], "output.log"), "w")
        sys.stdout = log_file

    instance_type = args['instance_type']

    start = time.time()
    if(instance_type=='polygon_tower_range'):
        S, A, S_id_dic, Phi_dic, pi = generate_polygon_tower(args['s_begin'], args['s_end'])
    elif(instance_type=='pick_the_diamond'):
        S, A, S_id_dic, Phi_dic, pi = generate_pick_the_diamond(args['diamond_edges'], args['num_slots'])
    elif(instance_type=='grid_world_navigation'):
        S, A, S_id_dic, Phi_dic, pi = generate_grid_world_navigation(args['grid_size'], args['repeat_moves'], args['feature_type'])
    else:
        raise ValueError("Wrong instance type provided!")

    print("State Space Size : ", len(S))
    print("Action Space Size : ", len(A))
    print("Feature Vector Size : ", len(Phi_dic[list(Phi_dic.keys())[0]]))
    print("# Feature Vectors : ", len(Phi_dic))
    
    # optimal teaching set, extreme rays unit vectors
    optimal_teaching_set, hat_E, Psi_dic = find_optimal_teaching_set(S, A, Phi_dic, pi)
    
    print("# Feature Diff Vectors : ", len(Psi_dic))
    print("="*75)
    print("Optimal Teaching Set size : ", len(optimal_teaching_set))
    print("Optimal Teaching Set (State ID list) : ", optimal_teaching_set)
    
    end = time.time()
    optimal_teaching_time = end-start

    for state_id in optimal_teaching_set:
        print_state(S_id_dic[state_id], instance_type)
        print('Optimal Action ', pi[state_id])
    
    ## randomized teaching
    start = time.time()
    num_iterations = args['num_randomized_trials']
    avg_draws, all_random_teaching_subsets = find_random_teaching_set(Psi_dic, num_iterations, hat_E)

    end = time.time()
    random_teaching_time = end-start

    random_teaching_set_size = np.array([len(drawn_subsets) for drawn_subsets in all_random_teaching_subsets])
    print("Random Teaching Set Size - Mean : {0:2.2f}, Std : {1:2.2f} ".format(np.mean(random_teaching_set_size), np.std(random_teaching_set_size)))
    
    # train consistent learners and evaluate them

    # Create the binary classification dataset D using teaching set
    X = np.array([Psi_dic[(s_id, a)] for s_id in optimal_teaching_set for a in A if a != pi[s_id]])
    X = np.vstack((X, -X))

    y = np.ones(len(optimal_teaching_set*(len(A)-1))) 
    y = np.concatenate([y, -y])

    for classifier_name in ['svm', 'perceptron', 'logistic']:
        classifier = train_classifier(X, y, classifier_name)
        error = evaluate_classifier(classifier, S, A, Phi_dic, pi)
        print(classifier_name, " error : ", error)


    return S, A, S_id_dic, Phi_dic, pi, Psi_dic, hat_E,\
          optimal_teaching_set, all_random_teaching_subsets, random_teaching_set_size, optimal_teaching_time, random_teaching_time

def run_code_parallel(args):
    
    if(args['terminal_output_to_log_file']):
        old_stdout = sys.stdout
        log_file = open(os.path.join(args['run_output_path'], "output.log"), "w")
        sys.stdout = log_file

    args_list = []
    main_run_output_path = args['run_output_path']

    instance_type = args['instance_type']
    if(instance_type=='polygon_tower_range'):
        for s_end in args['s_end_list']:
            cargs = args.copy()
            cargs['s_end'] = s_end
            cargs['run_output_path'] = os.path.join(main_run_output_path, 's_end_'+str(s_end))
            args_list.append(cargs)

    elif(instance_type=='pick_the_diamond'):
        for num_slots in args['num_slots_list']:
            cargs = args.copy()
            cargs['num_slots'] = num_slots
            cargs['run_output_path'] = os.path.join(main_run_output_path, 'num_slots_'+str(num_slots))
            args_list.append(cargs)

    elif(instance_type=='grid_world_navigation'):
        for grid_size in args['grid_size_list']:
            cargs = args.copy()
            cargs['grid_size'] = grid_size
            cargs['run_output_path'] = os.path.join(main_run_output_path, 'grid_size_'+str(grid_size))
            args_list.append(cargs)
    else:
        raise ValueError("Wrong instance type!")

    total_runs = len(args_list)
    batch_start_index_list = np.arange(0, total_runs, args['num_jobs_in_parallel'])

    all_output_list = []
    for batch_id, batch_start_index in enumerate(batch_start_index_list):
        print('*'*75)
        current_index_range = str(batch_start_index)+':'+str(batch_start_index+args['num_jobs_in_parallel'])
        print("Current process batch id : {0:2d}, batch index range : {1}".format(batch_id, current_index_range))
        print('*'*75)
        
        output_list = Parallel(n_jobs=args['num_jobs_in_parallel'])(delayed(run_code)(args) for args in args_list[batch_start_index:batch_start_index+args['num_jobs_in_parallel']])
        all_output_list.extend(output_list)
    
    # Open a pickle file for writing
    output_dump_file = os.path.join(main_run_output_path, 'all_output_list.pkl')

    updated_output_list = []
    if(args['save_output']=='full'):
       updated_output_list = all_output_list

    elif(args['save_output']=='partial'):
        for output_list in all_output_list:
            S, A, S_id_dic, _, pi, _, _, opt_teach_set, _, rand_teach_set_size, opt_teach_time, rand_teach_time = output_list
            updated_output_list.append((S, A, S_id_dic, pi, opt_teach_set, rand_teach_set_size, opt_teach_time, rand_teach_time))

    elif(args['save_output']=='only_size'):
        for output_list in all_output_list:
            _, _, _, _, _, _, _, opt_teach_set, _, rand_teach_set_size, _, _ = output_list
            updated_output_list.append((len(opt_teach_set), np.mean(rand_teach_set_size), np.std(rand_teach_set_size)))
    else:
        pass
    
    if(instance_type=='pick_the_diamond'):
        updated_output_list.append(args['num_slots_list'])
    elif(instance_type=='polygon_tower_range'):
        updated_output_list.append(args['s_end_list'])
    elif(instance_type=='grid_world_navigation'):
        updated_output_list.append(args['grid_size_list'])
    else:
        raise ValueError('Wrong instance type!')
    
    with open(output_dump_file, "wb") as f:
        pickle.dump(updated_output_list, f)
    f.close()



if __name__ == '__main__':
    
    stream = open('params.yml', 'r')
    args = yaml.full_load(stream)

    args['project_base_path'] = os.path.abspath(args['project_dir'])
    
    #################################################################
    #  - create a run_summary file for the first time in a project
    #
    #   initialize_run_summary()
    #################################################################
   
    set_output_directories_and_path(args)
    print(args)

    if(args['run_parallel']):
        run_code_parallel(args)
    else:
        run_code(args)
    