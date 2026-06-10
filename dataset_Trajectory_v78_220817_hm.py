# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:58:45 2022

@author: hossam.m
"""

import random
import logging
import math
import numpy
import json
import os
import concurrent.futures
import tensorflow as tf


class Dataset:
    def get_batch_count(self):
        raise NotImplementedError()
        

class StaticDataset(Dataset):
    def __init__(self,subsets,future_length):
        self.subsets=subsets
        self.batch_count=sum(((subset[2]+subset[1]-1)//subset[1] for subset in subsets))
        self.scales=numpy.linspace(1/future_length,1,future_length,dtype=numpy.float32)
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        def fetch_batch():
            if self.batch_index<self.batch_count:
                return preprocess_batch(self.batchs[self.batch_index],self.scales)
            else:
                return None
        self.fetch_batch=fetch_batch
    def get_batch_count(self):
        return self.batch_count
    def __iter__(self):
        logging.info('Batching')
        self.batch_index=0
        self.batchs=[]
        for subset,batch_size,_ in self.subsets:
            samples=[]
            for subsubset,subsubset_size_per_batch in subset:
                if len(subsubset)>subsubset_size_per_batch:
                    samples.extend(random.sample(subsubset,subsubset_size_per_batch))
                else:
                    samples.extend(subsubset)
            random.shuffle(samples)
            # samples.sort(key=lambda sample:len(sample.get_feature_length()))
            self.batchs.extend(collect_batch(samples,batch_size))
        random.shuffle(self.batchs)
        self.future=self.executor.submit(self.fetch_batch)
        assert self.batch_count==len(self.batchs), f'Batch size {self.batch_count} != {len(self.batchs)}'
        logging.info('Created %d batchs',self.batch_count)
        return self
    def __next__(self):
        batch=self.future.result()
        if batch:
            self.batch_index+=1
            self.future=self.executor.submit(self.fetch_batch)
            return batch
        else:
            raise StopIteration()
            
            
def split(string,delimiter):
    return string.split(delimiter) if string else []

feature_dimension=5

HALF_RIGHT_ANGLE=math.radians(45)
NORMALIZE_EVENT_COUNT=5
TWO_PI=2*numpy.pi

def normalize(start_index,trace,past_length,future_length,add_noise=False):
    segment=trace[start_index:start_index+past_length+future_length]
    normal_direction=segment[past_length-1,:2]-segment[past_length-1-NORMALIZE_EVENT_COUNT,:2]
    difference=segment[1:]-segment[:-1]
    dxy=difference[:,:2]
    dtime=difference[:,2:3]

    speed = []
    for i in range(len(dtime)):
        if dtime[i] == 0:
            dtime[i] = 4
        speed_i = [numpy.sqrt((dxy[i,0]**2 + dxy[i,1]**2))/dtime[i]]
        speed.append(speed_i)

    speed = numpy.array(speed)
    speed = speed.reshape(speed.shape[0], speed.shape[1])
    # print(dxy.shape)

    # print(speed.shape)
    dp=difference[:,3:4]
    dt=difference[:,4:5]
    do=difference[:,5:]
    if add_noise:
        dxy=dxy*(1+0.10*numpy.random.normal())
        # angle=numpy.random.uniform(0,2*numpy.pi)
        # cs = math.cos(angle)
        # sn = math.sin(angle)
        # transformation=numpy.array([[cs,sn],[-sn,cs]])*(1+0.10*numpy.random.normal())
        # dxy=numpy.dot(dxy,transformation)
    angle=math.atan2(normal_direction[0], normal_direction[1])+HALF_RIGHT_ANGLE
    cs = math.cos(angle)
    sn = math.sin(angle)
    dxy=numpy.dot(dxy,numpy.array([[cs,sn],[-sn,cs]]))
    do=do-numpy.round(do/TWO_PI)*TWO_PI
    smoothed_speed = speed
    smoothed_speed = smoothed_speed.reshape(speed.shape[0], speed.shape[1])
    # print(smoothed_speed)
    # print(smoothed_speed.shape)
    # print(dxy.shape)
    feature=numpy.hstack([dxy])[:past_length-1]
    caption=numpy.zeros((future_length,2),dtype=numpy.float32)
    dxy=dxy[past_length-1:]
    caption[:dxy.shape[0]]=dxy
    caption=numpy.cumsum(caption,axis=0)
    return feature,caption


def normalize2(start_index,trace,past_length,future_length):
    segment=trace[start_index:start_index+past_length+future_length]
    feature = segment[:past_length-1,[0,1,3,4,5]]
    caption = segment[past_length:, :2]
    return feature,caption



class Expression:
    def get_feature_caption(self):
        raise NotImplementedError()

class ParsedExpression(Expression):
    def __init__(self,start_index,trace,past_length,future_length):
        self.feature,self.caption=normalize(start_index,trace,past_length,future_length)
    def get_feature_caption(self):
        return self.feature,self.caption

class UnparsedExpression(Expression):
    def __init__(self,start_index,trace,past_length,future_length):
        self.start_index=start_index
        self.trace=trace
        self.past_length=past_length
        self.future_length=future_length
    def get_feature_caption(self):
        return normalize(self.start_index,self.trace,self.past_length,self.future_length,True)
MAX_HISTORY_LENGTH=1000000
MIN_PAST_LENGTH=3

def load_samples(index_file,dataset_path,past_length,future_length,cache_feature=True,max_history_length=MAX_HISTORY_LENGTH):
    with open(os.path.join(dataset_path,index_file)) as lines:
        logging.info('Parsing dataset %s',index_file)
        traces=[]
        trace=[]
        for line in lines:
            fields=[float(field) for field in split(line.rstrip('\n'),' ')]
            if fields:
                trace.append(fields)
                if len(trace)>=max_history_length:
                    print(f'Splitting trace longer than {max_history_length}')
                    traces.append(numpy.array(trace,dtype=numpy.float32))
                    trace=trace[-max_history_length//2:]
            elif trace:
                if len(trace)>=MIN_PAST_LENGTH:
                    trace=numpy.array(trace,dtype=numpy.float32)
                    padding=numpy.tile(trace[0],(past_length-MIN_PAST_LENGTH,1))
                    traces.append(numpy.vstack((padding,trace)))
                trace=[]
        logging.info('Parsed %d traces',len(traces))
        samples=[]
        for trace in traces:
            for i in range(trace.shape[0]-past_length+1):
                if numpy.max(numpy.abs(trace[i+past_length-1,:2]-trace[i+past_length-1-NORMALIZE_EVENT_COUNT,:2]))>1e-7:
                    samples.append(ParsedExpression(i,trace,past_length,future_length) if cache_feature else UnparsedExpression(i,trace,past_length,future_length))
        logging.info('Parsed %d samples',len(samples))
        return samples


def load_traces(index_file,dataset_path,past_length,future_length,cache_feature=True,max_history_length=MAX_HISTORY_LENGTH):
    with open(os.path.join(dataset_path,index_file)) as lines:
        logging.info('Parsing dataset %s',index_file)
        traces=[]
        trace=[]
        for line in lines:
            fields=[float(field) for field in split(line.rstrip('\n'),' ')]
            if fields:
                trace.append(fields)
                if len(trace)>=max_history_length:
                    print(f'Splitting trace longer than {max_history_length}')
                    traces.append(numpy.array(trace,dtype=numpy.float32))
                    trace=trace[-max_history_length//2:]
            elif trace:
                if len(trace)>=MIN_PAST_LENGTH:
                    trace=numpy.array(trace,dtype=numpy.float32)
                    padding=numpy.tile(trace[0],(past_length-MIN_PAST_LENGTH,1))
                    traces.append(numpy.vstack((padding,trace)))
                trace=[]
        logging.info('Parsed %d traces',len(traces))
        samples=[]
        segments=[]
        future_trajs = []
        for trace in traces:

            for i in range(trace.shape[0]-past_length+1):
                segment=trace[i:i+past_length]
                future_traj=trace[i+past_length+1:i+past_length+1+future_length][:,0:2]
                segments.append(segment)
                future_trajs.append(future_traj)

                if numpy.max(numpy.abs(trace[i+past_length-1,:2]-trace[i+past_length-1-NORMALIZE_EVENT_COUNT,:2]))>1e-7:
                    samples.append(ParsedExpression(i,trace,past_length,future_length) if cache_feature else UnparsedExpression(i,trace,past_length,future_length))
        logging.info('Parsed %d samples',len(samples))
        return traces, segments, future_trajs






def preprocess_batch(expressions,scales):
    feature_caption=[sample.get_feature_caption() for sample in expressions]
    x=numpy.array([feature for feature,_ in feature_caption],dtype=numpy.float32)
    y=numpy.array([caption for _,caption in feature_caption],dtype=numpy.float32)
    return x,y,numpy.tile(scales,(len(feature_caption),1))


def collect_batch(samples,batch_size):
    return [samples[i:i+batch_size] for i in range(0,len(samples),batch_size)]


def load_train_subset(descriptor,base_path,past_length,future_length):
    logging.info('Loading train subset')
    subsubsets=[]
    sample_count_keep,sample_count_skip,sample_count_per_batch=0,0,0
    for subsubset_descriptor in descriptor['index']:
        if isinstance(subsubset_descriptor,str):
            index_file=subsubset_descriptor
            cache_feature=False
            subsubset_size_per_batch=-1
            max_history_length=MAX_HISTORY_LENGTH
        else:
            index_file=subsubset_descriptor['file']
            cache_feature=subsubset_descriptor['cache_feature'] if 'cache_feature' in subsubset_descriptor else False
            subsubset_size_per_batch=subsubset_descriptor['size_per_batch'] if 'size_per_batch' in subsubset_descriptor else -1
            max_history_length=subsubset_descriptor['max_history_length'] if 'max_history_length' in subsubset_descriptor else MAX_HISTORY_LENGTH
        samples=[sample for sample in load_samples(index_file,base_path,past_length,future_length,cache_feature,max_history_length)]
        subsubset_sample_count=len(samples)
        subsubset_sample_count_keep=len(samples)
        sample_count_keep+=subsubset_sample_count_keep
        sample_count_skip+=(subsubset_sample_count-subsubset_sample_count_keep)
        subsubset_size_per_batch=min(subsubset_sample_count_keep,subsubset_size_per_batch) if subsubset_size_per_batch>0 else subsubset_sample_count_keep
        sample_count_per_batch+=subsubset_size_per_batch
        subsubsets.append((samples,subsubset_size_per_batch))
    # supplements=descriptor['supplement_index'] if 'supplement_index' in descriptor else None
    logging.info('Loaded %d train samples with batch size %d (Skiped %d), %d per batch',sample_count_keep,descriptor['batch_size'],sample_count_skip,sample_count_per_batch)
    return (subsubsets,descriptor['batch_size'],sample_count_per_batch)

def load_test_subset(descriptor,base_path,past_length,future_length):
    logging.info('Loading test/valid set')
    samples=[sample for index_file in descriptor
                   for sample in load_samples(index_file,base_path,past_length,future_length)]
    logging.info('Loaded %d test/valid samples',len(samples))
    return samples

def load_data(description_file,include_train,include_valid,include_test):
    with open(description_file) as lines:
        descriptor=json.load(lines)
        base_path=os.path.dirname(description_file)
        past_length=descriptor['past_length']
        future_length=descriptor['future_length']
        splits={}
        if include_train:
            splits['train_set']=[load_train_subset(train_descriptor,base_path,past_length,future_length)
                                 for train_descriptor in descriptor['train_set']]
        if include_valid:
            splits['valid_set']=load_test_subset(descriptor['valid_set'],base_path,past_length,future_length)
        if include_test:
            splits['test_set']=load_test_subset(descriptor['test_set'],base_path,past_length,future_length)
        return splits,future_length
                