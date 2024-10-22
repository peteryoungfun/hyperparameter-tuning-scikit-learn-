#############################
"""
Expected input for opt_without_label function:
func: the target function
c_list: continuous parameter list and feasible space. Expected form: [[continuous_para1,[start,end]],[continuous_para2,[start,end]],...]
i_list: int parameter list and feasible space. Expected form: [[int_para1,[feasible list]],[int_para2,[feasible list],...]
label: label parameter which chose to be fixed. If there's no label parameter, label=[] should be used.
alpha: float, step size, which will be used to calculate neighbors for continuous parameter
max_iter: int, maximum iteration
min_improve: float, minimum improve which will be accepted to continue iteration
Expected input for opt_neighbor function:
func: the target function
continuous_list: continous parameter list and feasible space. Expected form: [[continuous_para1,[start,end]],[continuous_para2,[start,end]],...]
int_list: int parameter list and feasible space. Expected form: [[int_para1,[feasible list]],[int_para2,[feasible list],...]
label_list: label parameter list and feasible space. Expected form: [[label_para1,[feasible list]],[label_para2,[feasible list],...]
alpha: float, step size, which will be used to calculate neighbors for continuous parameter
max_iter: int, maximum iteration
min_improve: float, minimum improve which will be accepted to continue iteration
random_test_num: int, number of randomly initialized points in each label condition
"""
#############################

import itertools
import random
import warnings
warnings.filterwarnings('ignore')
#define gradient descent for continuous parameters
#calculate approximate gradient here
def gd(func,key_w_c,para_dict,gra_dist,label):
    #print(para_dict)
    p_dict=dict(para_dict)
    p_dict[key_w_c]-=gra_dist
    value1=func(**para_dict,**label)
    value2=func(**p_dict,**label)
    gradient=(value1-value2)/gra_dist
    return gradient
    
#define opt function without label parameter
def opt_without_label(func,c_list,i_list,label,alpha,max_iter,min_improve):
    #check if all 3 lists are empty lists. if yes then algorithm will not initialize
    if len(c_list)+len(i_list)+len(label)==0:
        return ("No parameter input")
    
    #if there's only label inputs, we only calculate once
    elif len(label)!=0 and len(c_list)+len(i_list)==0:
        return [[label,func(**label)],1]
    
    ##if there's continous or int input, algorithm will start
    else:
        #initialization
        convergence=False
        n_iter=0
        #randomly assign initialized value from feasible space
        print('initialized parameter value:')
        para_dict={}
        for i in range(len(c_list)):
            x=random.uniform(int(c_list[i][1][0]),int(c_list[i][1][1]))
            para_dict[c_list[i][0]]=x
        for j in range(len(i_list)):
            y=random.choice(i_list[j][1])
            para_dict[i_list[j][0]]=y
        print(para_dict)
        #calculate the initialized function value and create exist_path, saving calculated parameters to avoid duplicated calculations.
        exist_path=[]
        i_value=func(**para_dict,**label)
        print("initialized function value:",i_value)
        exist_path.append(para_dict)
        current_paraset=dict(para_dict)
        current_value=i_value
        #iteration start
        #calculate function call time (function calculation times)
        n_c=1
        while(convergence==False):
            neigh_point=[]
            parameter_list=[]
            possible_combine_lists=[]
            #continuous: apply gradient descent for continuous parameter
            for i in range(len(c_list)):
                parameter_list.append(c_list[i][0])
                stp=gd(func,c_list[i][0],current_paraset,0.00001,label)*alpha
                n_c+=2
                prev=current_paraset[c_list[i][0]]-stp
                afte=current_paraset[c_list[i][0]]+stp
                cur_list=[prev,current_paraset[c_list[i][0]],afte]
                #check if those points are out of boundary. If yes then remove from neighbor points list.
                for unit in cur_list:
                    if unit < c_list[i][1][0] or unit > c_list[i][1][1]:
                        cur_list.remove(unit)
                possible_combine_lists.append(cur_list)
            #int
            for i in range(len(i_list)):
                parameter_list.append(i_list[i][0])
                if i_list[i][1]==1:
                    cur_list=[current_paraset[i_list[i][0]]]
                else:
                    cur_point=i_list[i][1].index(current_paraset[i_list[i][0]])
                    if cur_point == 0:
                        cur_list=[current_paraset[i_list[i][0]],i_list[i][1][1]]
                    elif cur_point == len(i_list[i][1])-1:
                        cur_list=[i_list[i][1][-2],current_paraset[i_list[i][0]]]
                    else:
                        cur_list=[i_list[i][1][cur_point-1],i_list[i][1][cur_point],i_list[i][1][cur_point+1]]
                possible_combine_lists.append(cur_list)
            #combine possible parameter outcomes
            allpc=list(itertools.product(*possible_combine_lists))
            for i in range(len(allpc)):
                this_point={}
                for j in range(len(parameter_list)):
                    this_point[parameter_list[j]]=allpc[i][j]
                neigh_point.append(this_point)
            #test if neighbor already calculated before, if yes then delete
            for point in neigh_point:
                #test if already calculated
                if point in exist_path:
                    neigh_point.remove(point)
            #apply neighbor parameter test and choose the best group of parameters as opt_parameters in this iteration
            max=[]
            for point in neigh_point:
                value=func(**point,**label)
                n_c+=1
                exist_path.append(point)
                if max==[]:
                    max=[point,value]
                else:
                    if value > max[1]:
                        max=[point,value]

            #if there's no neighbor function value better than current or the change is too small
            #iteration over
            if max[1]>current_value+min_improve and n_iter < max_iter:
              current_paraset=max[0]
              current_value=max[1]
              print(f"{n_iter+1}th iteration ------- value: {current_value}")
              n_iter+=1
            
            elif n_iter >=max_iter:
              convergence=True
              print(f"Maximum number of iterations {max_iter} reached!")

            elif max[1]>current_value:
                convergence=True
                print(f"Converged because of lower improvement than limitation({min_improve})!")

            else:
                convergence=True
                print("Converged because of local maximum reached!")

        print(f'this round we have {n_c} calculations')
        return [[current_paraset,current_value],n_c]



#define whole algorithm based on previous one
def opt_neighbor(func,continuous_list,int_list,label_list,alpha,max_iter,min_improve,random_test_num):
    #check if all 3 lists are empty lists. if yes then algorithm will not initialize
    if len(continuous_list)+len(int_list)+len(label_list)==0:
        return ("No parameter input!")
    else:
        combine_list=[]
        label_name=[]
        labels=[]
        for i in range(len(label_list)):
            combine_list.append(label_list[i][1])
            label_name.append(label_list[i][0])
        alllc=list(itertools.product(*combine_list))
        for i in range(len(alllc)):
            label_point={}
            for j in range(len(label_name)):
                label_point[label_name[j]]=alllc[i][j]
            labels.append(label_point)
        total_calculation=0
        final_output=[]
        if labels==[]:
            print("Run algorithm without label parameter")
            while (n<random_test_num):
                print("---------------------------New Random Initialization------------------------------")
                print(f"For {n+1}st random initialization:")
                cur=opt_without_label(func,continuous_list,int_list,labels,alpha,max_iter,min_improve)
                total_calculation+=cur[1]
                if n==0 or cur[0][1]>best[1]:
                    best=cur[0]
                n+=1
            final_output.append((label,best))
            final_output_3=sorted(final_output,key=lambda x: x[1][1],reverse=True)[:3]
        else:
            for label in labels:
                n=0
                print("**************************New Label Parameter***********************************")
                print(f"For label parameter {label}:")
                while (n<random_test_num):
                    print("---------------------------New Random Initialization------------------------------")
                    print(f"For {n+1}st random initialization:")
                    cur=opt_without_label(func,continuous_list,int_list,label,alpha,max_iter,min_improve)
                    total_calculation+=cur[1]
                    if n==0 or cur[0][1]>best[1]:
                        best=cur[0]
                    n+=1
                final_output.append((label,best))
            final_output_3=sorted(final_output,key=lambda x: x[1][1],reverse=True)[:3]
        print(f"Total function calculation(including gradient approximate calculation and neighbor points calculation in iterations):{total_calculation}")
        return final_output_3
