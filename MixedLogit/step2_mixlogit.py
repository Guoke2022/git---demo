
from xlogit import MultinomialLogit,MixedLogit
import pandas as pd
import os
import warnings
import logging
import json
import time
from functools import wraps
import numpy as np
import scipy.optimize as opt
import seaborn as sns
import  matplotlib.pyplot as plt
import matplotlib as mpl
import math
import time
from datetime import datetime
import gc
import cupy as cp
import random



# -----------Plot style settings------------------
plt.rcParams["font.size"] = 14


# ------------Log decorator-----------------------
def process_log(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        try:
            self.logger.info(f"Start calling the function: {func.__name__}")
            start_time = time.time()
            result = func(self,*args, **kwargs)
            end_time = time.time()
            self.logger.info(f"Function {func.__name__}，time: {end_time - start_time:.4f} 秒 \n")
            return result
        except Exception as e:
            self.logger.error(f"Function {func.__name__} has error: {e}")
            raise
    return wrapper


# ------------DiscreteChoiceExperiment-----------------
class DiscreteChoiceExperiment():
    def __init__(self,config:dict):
        """
        Initialize parameters
        """

        # Load parameters
        self.experiment_name = config["experiment_name"]
        self.df_path = config["df_path"]
        self.group_attribute = config["group_attribute"]
        self.batch_size = config["batch_size"]
        self.model_type = config["model_type"]
        self.random_coefficient = config["random_coefficient"]
        self.sampling_rate =  config["sampling_rate"]

        self.use_cols = []
        self.avail = config["avail_col"]
        self.id_col = config["id_col"]
        self.alt_col = config["alt_col"]
        self.choice_col = config["choice_col"]
        self.explanatory_variable_list = config["explanatory_variable_list"]

        self.quality_utility_col = config["quality_utility_col"]
        self.cost_utility_col = config["cost_utility_col"]

        # Merge
        self.use_cols.append(self.avail)
        self.use_cols.append(self.id_col)
        self.use_cols.append(self.alt_col)
        self.use_cols.append(self.choice_col)
        self.use_cols.extend(self.explanatory_variable_list)

        if self.group_attribute:
            self.use_cols.append(self.group_attribute)

        # File directory
        self.work_dir = f"./{self.experiment_name}"
        self.config_dir = f"{self.work_dir}/config"
        self.log_dir = f"{self.work_dir}/log"
        self.result_dir = f"{self.work_dir}/result"
        self.WTT_dir = f"{self.work_dir}/WTT"
        

        # Create initialization path folder
        os.makedirs(self.work_dir,exist_ok=True)
        os.makedirs(self.config_dir,exist_ok=True)
        os.makedirs(self.log_dir,exist_ok=True)
        os.makedirs(self.result_dir,exist_ok=True)
        os.makedirs(self.WTT_dir,exist_ok=True)


        # Set up logging
        logging.basicConfig(
            filename=f"{self.log_dir}/experiment_{datetime.now().strftime('%Y%m%d_%H')}.log",  
            level=logging.INFO,  
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
            force=True
        )
        self.logger = logging.getLogger()

        # Save config 
        with open(f"{self.config_dir}/config.json","w") as json_file:
            json.dump(config,json_file,indent=4)

        # Load data and initialize the model
        self.df = self._load_df()
        self.model = self._create_model()
        self.result_df = pd.DataFrame()
        
        self.null_ll = None
        self.pseudo_r_2 = None
        self.wtt = None


    # ------------Process function----------------
    @ process_log
    def _load_df(self):
        """
        Load and sample data
        """
        df = pd.read_parquet(self.df_path)

        ids = df["id"].unique()
        ids = ids.tolist()
        ids_len = len(ids)

        sampled_ids = random.sample(ids,k=int(ids_len*self.sampling_rate))
        df = df[df["id"].isin(sampled_ids)]

        return df

            
    @ process_log
    def _create_model(self):
        """
        Create the model 
        """

        # MultinomialLogit model
        if self.model_type=="MultinomialLogit":
            return MultinomialLogit()
        
        # MixedLogit model
        if self.model_type=="MixedLogit":
            return MixedLogit()
        
        
    @ process_log
    def reset(self):
        """
        Reset
        """
        self.model = self._create_model()
        self.result_df = pd.DataFrame()
        self.null_ll = None
        self.pseudo_r_2 = None
        self.wtt = None
            

    # ------------Experimental objective function----------------
    @ process_log
    def fit(self,df:pd.DataFrame):

        """
        Model estimation
        """

        # MultinomialLogit estimation
        if self.model_type=="MultinomialLogit":
            self.model.fit(X=df[self.explanatory_variable_list],
                        y=df[self.choice_col],
                        varnames=self.explanatory_variable_list,
                        ids=df[self.id_col],
                        alts=df[self.alt_col],
                        )

        # MixedLogit estimation
        if self.model_type=="MixedLogit":
            self.model.fit(X=df[self.explanatory_variable_list],
            y=df[self.choice_col],
            varnames=self.explanatory_variable_list,
            ids=df[self.id_col],
            alts=df[self.alt_col],
            batch_size=self.batch_size,
            avail=df[self.avail],
            n_draws=100,
            randvars=self.random_coefficient)

    @ process_log
    def calculate_null_ll(self,df:pd.DataFrame):
        """
        Null model likelihood
        """

        choice_num = len(df[self.id_col].unique())
        alt_num = len(df[self.alt_col].unique())
        null_ll = choice_num*np.log(1/alt_num)
        return null_ll
    
    @ process_log 
    def calculate_pseudo_r_2(self):
        """
        Compute pseudo R-squared
        """
        self.pseudo_r_2 = 1-self.model.loglikelihood/self.null_ll

    @ process_log
    def calculate_wtt(self,df:pd.DataFrame,save_name:str):
        """
        Calculate WTT
        """

        # Sampling from the distribution range of distance x

        percentiles = np.linspace(0, 1, 21)
        df = df[df[self.cost_utility_col[0]]>0]
        x = df[self.cost_utility_col[0]].quantile(percentiles)

        # Distance coefficient
        a, b, c = 0, 0, 0
        for i in range(len(self.model.coeff_)):
            if self.model.coeff_names[i] == self.cost_utility_col[0]:
                a = self.model.coeff_[i]
            if self.model.coeff_names[i] == self.cost_utility_col[1]:
                b = self.model.coeff_[i]
            if self.model.coeff_names[i] == self.cost_utility_col[2]:
                c = self.model.coeff_[i]

        # Quality utility coefficient
        quality_utility_coeff_df = pd.DataFrame()
        for i in range(len(self.model.coeff_)):
            for col in self.quality_utility_col:
                print(col)
                print(self.model.coeff_names[i])
                if self.model.coeff_names[i]==col:
                    quality_utility_coeff_df[col] = [self.model.coeff_[i]]
        
        print(quality_utility_coeff_df)

        # Compute WTT
        self.wtt = pd.DataFrame({self.cost_utility_col[0]:x})
        for col in quality_utility_coeff_df.columns:
            y = []
            quality_utility_coeff = quality_utility_coeff_df.loc[0,col]
            for value in x:
                y.append(-quality_utility_coeff/(a+ 2*b*value + 3*c*(value**2)))
            self.wtt[col]=y

        # Save WTT
        os.makedirs(f"{self.WTT_dir}/{save_name}",exist_ok=True)
        os.makedirs(f"{self.WTT_dir}/{save_name}/df",exist_ok=True)
        self.wtt.to_csv(f"{self.WTT_dir}/{save_name}/df/result.csv")

        return self.wtt

    @ process_log
    def plot_wtt(self,save_name:str):
        """
        # Plot the distribution of WTT
        """
        plt_num = len(self.wtt.columns[1:])
        fig_col = 3
        fig_row = math.ceil(plt_num / fig_col) 
        fig, axes = plt.subplots(fig_row, fig_col, figsize=(6*fig_col, 5*fig_row))
        axes = axes.flatten()
        palette = sns.color_palette("viridis", len(self.wtt.columns[1:]))

        for i in range(0,plt_num):
            color = palette[i]
            title_name = f"WTT" + " " + self.wtt.columns[i+1].split("_")[0]+ " " +self.wtt.columns[i+1].split("_")[1]

            sns.lineplot(data=self.wtt,
                         x=self.wtt.columns[0],
                         y=self.wtt.columns[i+1],
                         ax=axes[i],
                         color=color,
                         lw=2,
                         marker="o",
                         markersize=0)
            
            axes[i].scatter(self.wtt[self.wtt.columns[0]],self.wtt[self.wtt.columns[i+1]],s=60,color=color)
            axes[i].axhline(y=0, color='gray', linewidth=1, linestyle='--')
            axes[i].set_title(title_name)
            axes[i].set_xlabel("Option distance(100km)",fontsize=14)
            axes[i].set_ylabel("WTT(100km)",fontsize=14)

        os.makedirs(f"{self.WTT_dir}/{save_name}/plt",exist_ok=True)
        plt.savefig(f"{self.WTT_dir}/{save_name}/plt/wtt.jpg")
        plt.tight_layout()
        plt.close()

    @ process_log
    def record_result(self,
                      individual_count:int):
        """
        Record the estimated parameter results
        """

        self.pseudo_r_2 = 1-self.model.loglikelihood/self.null_ll

        fit_info_cols = {"Message":[self.model.estimation_message],
                                    "Iterations":[self.model.total_iter],
                                    "Function evaluations":[self.model.total_fun_eval],
                                    "Estimation time":[self.model.estim_time_sec],
                                    "Log-Likelihood":[self.model.loglikelihood],
                                    "AIC":[self.model.aic],
                                    "BIC":[self.model.bic],
                                    "null_ll":[self.null_ll],
                                    "pseudo_r_2":[self.pseudo_r_2],
                                    "individual_count":individual_count}

        for col,info in fit_info_cols.items():
            self.result_df[col]=info
        
        coeff_data = {}
        for i, name in enumerate(self.model.coeff_names):
            coeff_data[f"{name}_coeff"] = [self.model.coeff_[i]]
            coeff_data[f"{name}_stderr"] = [self.model.stderr[i]]
            coeff_data[f"{name}_zvalues"] = [self.model.zvalues[i]]
            coeff_data[f"{name}_pvalues"] = [self.model.pvalues[i]]
        coeff_df = pd.DataFrame(coeff_data)
        self.result_df = pd.concat([self.result_df, coeff_df], axis=1)

    @ process_log
    def save_record_result(self,save_path):
        """
        Save the recorded results
        """
        self.result_df.to_json(save_path,orient='records', indent=4)


    # ------------Main procedure execution function of the experiment----------------
    @ process_log
    def run_experiment(self):

        """
        Main experimental procedure, with and without grouping
        """

        # Group
        if self.group_attribute:
            for group_class in self.df[self.group_attribute].unique():
                group_df = self.df[self.df[self.group_attribute]==group_class].copy()
                self.fit(group_df)
                self.null_ll = self.calculate_null_ll(group_df)
                self.calculate_wtt(group_df,group_class)
                self.plot_wtt(group_class)

                individual_count = len(group_df[self.id_col].unique())
                group_result_path = self.result_dir + f"/{group_class}.json"
                self.record_result(individual_count)
                self.save_record_result(group_result_path)

                self.reset()

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        # No group
        else:

            self.fit(self.df)
            self.null_ll = self.calculate_null_ll(self.df)
            self.calculate_wtt(self.df,"no_group")
            self.plot_wtt("no_group")

            individual_count = len(self.df[self.id_col].unique())
            result_path = self.result_dir + f"/result.json"
            self.record_result(individual_count)
            self.save_record_result(result_path)
            
            self.reset()


if __name__=="__main__":

    # Group experiment
    config = {

        "experiment_name": "MixedLogit_without_interaction_no_group",

        "id_col":"id",
        "alt_col":"option_hosp_id",
        "choice_col":"judge",
        "avail_col":"CITY_JUDGE",
        "df_path":"data/process/all_city.parquet",
        "group_attribute":None,
        "sampling_rate":0.07,

        "model_type":"MixedLogit",
        "batch_size":5,
        "explanatory_variable_list":['option_grade',
                                     'option_beds',
                                     'option_rep',
                                     'option_distance_100_km',
                                     'option_distance_100_km_2',
                                     'option_distance_100_km_3',
                                     ],
        "random_coefficient":{'option_grade':'n','option_beds':'n','option_rep':'n',
                                     'option_distance_100_km':'n','option_distance_100_km_2':'n',
                                     'option_distance_100_km_3':'n'},

        "quality_utility_col":['option_grade','option_beds','option_rep'],
        "cost_utility_col":['option_distance_100_km','option_distance_100_km_2',
                                     'option_distance_100_km_3'] 
    }
    expriment = DiscreteChoiceExperiment(config)
    expriment.run_experiment()
    del expriment
    gc.collect()


    # No group experiment
    config = {

        "experiment_name": "MixedLogit_without_interaction_SES_group",

        "id_col":"id",
        "alt_col":"option_hosp_id",
        "choice_col":"judge",
        "avail_col":"CITY_JUDGE",
        "df_path":"data/process/all_city.parquet",
        "group_attribute":"SES_Level",
        "sampling_rate":0.07,

        "model_type":"MixedLogit",
        "batch_size":5,
        "explanatory_variable_list":['option_grade',
                                     'option_beds',
                                     'option_rep',
                                     'option_distance_100_km',
                                     'option_distance_100_km_2',
                                     'option_distance_100_km_3',
                                     ],
        "random_coefficient":{'option_grade':'n','option_beds':'n','option_rep':'n',
                                     'option_distance_100_km':'n','option_distance_100_km_2':'n',
                                     'option_distance_100_km_3':'n'},

        "quality_utility_col":['option_grade','option_beds','option_rep'],
        "cost_utility_col":['option_distance_100_km','option_distance_100_km_2',
                                     'option_distance_100_km_3'] 
    }
    expriment = DiscreteChoiceExperiment(config)
    expriment.run_experiment()
    del expriment
    gc.collect()
