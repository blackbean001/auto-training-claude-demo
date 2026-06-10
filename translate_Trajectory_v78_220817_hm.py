import dataset_Trajectory_v78_220817_hm
import cnn_gru_v7_Trajectory_v78_220817_hm
import pandas as pd
import logging
import time
import numpy
import math
import os
import json
import tensorflow as tf
from pandas import Series

def main():

    model_file = 'model/Trajectory_v78_220817_hm'
    

    
    dataset_descriptor_files = ["data/testing_fast_speed_big_curves_with276dpi_0503.json",
                                    "data/testing_fast_speed_chinese_characters_with276dpi_0503.json",
                                    "data/testing_fast_speed_polylines_with276dpi_0503.json",
                                    "data/testing_fast_speed_small_curves_with276dpi_0503.json",
                                    "data/testing_fast_speed_straight lines_with276dpi_0503.json",
                                    "data/testing_normal_speed_big_curves_0504_with276dpi.json",
                                    "data/testing_normal_speed_chinese_characters_with276dpi_0503.json",
                                    "data/testing_normal_speed_polylines_with276dpi_0503.json",
                                    "data/testing_normal_speed_small_curves_with276dpi_0503.json",
                                    "data/testing_normal_speed_straight_lines_with276dpi_0503.json"]
        
    
    results = {}
    
    for dataset_descriptor_file in dataset_descriptor_files:
                                                        
        splits,future_length=dataset_Trajectory_v78_220817_hm.load_data(dataset_descriptor_file,False,False,True)
        
        model, infer, options = cnn_gru_v7_Trajectory_v78_220817_hm.load_model_testing(model_file)
         
        
        def gen_sample(x):
            logits = None
            t = None
            weight_sum=0
            # for f_infer,options,weight in models:
            output,output_t = infer(x)
            if logits is not None:
                logits+=output
                t+=output_t
            else:
                logits=output
                t=output_t
                weight_sum+=1
            logits/=weight_sum
            t/=weight_sum
            return logits,t
        
        
        
        
        
        
        
        frequency=240
        tolerance=1.5
        test_set=splits['test_set']
        ud_epoch_start = time.time()
        display_frequency=500
        testBatchSize=512
        distance_err,angle_err,time_err,good_rate,tested_count=0,0,0,0,0
        batchCount=len(test_set)//testBatchSize
        scales=numpy.linspace(1/future_length,1,future_length,dtype=numpy.float32)
        time_var_list = []
        for i in range(batchCount):
            if i%display_frequency==0:
                print(f'{i}/{batchCount}')
            test_batch=test_set[i*testBatchSize:(i+1)*testBatchSize]
            feature_caption=[sample.get_feature_caption() for sample in test_batch]
            prediction,predicted_time=gen_sample(numpy.array([feature for feature,_ in feature_caption],dtype=numpy.float32))
            groundTruth=numpy.array([caption for _,caption in feature_caption],dtype=numpy.float32)
            time_to_predict=numpy.tile(scales,(len(feature_caption),1))
            distance_metric,angle_metric,time_metric=cnn_gru_v7_Trajectory_v78_220817_hm.calculate_loss(prediction,predicted_time,groundTruth,time_to_predict)
            
            distance_err+=numpy.sum(numpy.square(distance_metric.numpy()))
            angle_err+=numpy.sum(angle_metric.numpy())
            time_err+=numpy.sum(time_metric)
            time_var_list.append(time_metric)
            good_rate+=numpy.count_nonzero(distance_metric<tolerance)
            tested_count+=len(test_batch)
        
        distance_err/=tested_count
        distance_err=numpy.sqrt(distance_err)
        angle_err/=tested_count
        time_err/=tested_count
        good_rate/=tested_count
        time_var = numpy.sqrt(numpy.var(time_var_list))
        
        results[dataset_descriptor_file] = {}
        results[dataset_descriptor_file]['distance_err'] = distance_err/2
        results[dataset_descriptor_file]['angle_err'] = angle_err
        results[dataset_descriptor_file]['APT'] = (1-time_err)*future_length*1000/frequency
        results[dataset_descriptor_file]['APT_Standard_Dev'] = time_var

        print("printing " + dataset_descriptor_file)
        results_df  = Series(results,index=results.keys())
        results_df.to_csv('testing_results_Trajectory_v78_220817_hm.csv')


        
if __name__ == "__main__":
    main()        