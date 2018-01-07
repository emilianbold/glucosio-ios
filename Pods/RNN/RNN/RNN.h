//
//  RNN.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNNet.h"
#import "RNNPattern.h"
#import "RNNFetcher.h"
#import "RNNCost.h"
#import "RNNSequenceOutput.h"

@class RNN;

static NSInteger kRNNFullBPTT = 0;

typedef void(^RNNCompletion)(NSInteger totalIteration, RNN *network);
typedef void(^RNNIteration)(NSInteger iteration, RNN *network);
typedef void(^RNNPredication)(NSArray <RNNSequenceOutput *> *sequenceOutputs);
typedef void(^RNNBeforeUpdate)(NSInteger iteration, RNN *network);

@interface RNN : NSObject <NSCoding>

@property (nonatomic, readonly) NSMutableArray <RNNPattern *> *inputLayer;
@property (nonatomic, readonly) NSMutableArray <RNNNet *> *hiddenLayer;
@property (nonatomic, readonly) NSMutableArray <RNNNet *> *outputLayer;
@property (nonatomic) RNNNetActivations activation;
@property (nonatomic) double learningRate;
@property (nonatomic) double randomMax;
@property (nonatomic) double randomMin;
@property (nonatomic) double convergenceError;
@property (nonatomic) NSInteger maxIteration;
@property (nonatomic) NSInteger iteration;
@property (nonatomic) NSInteger timestepSize;

@property (nonatomic, copy) RNNCompletion trainingCompletion;
@property (nonatomic, copy) RNNIteration trainingIteraion;
@property (nonatomic, copy) RNNBeforeUpdate beforeUpdate;

@property (nonatomic, readonly) RNNCost *costFunction;
@property (nonatomic, readonly) RNNFetcher *fetcher;

+ (instancetype)sharedNetwork;
- (instancetype)init;

- (void)addHiddenNet:(RNNNet *)net;
- (void)addOutputNet:(RNNNet *)net;
- (void)addPatternWithFeatures:(NSArray <NSNumber *> *)features targets:(NSArray <NSNumber *> *)targets;
- (void)addPattern:(RNNPattern *)pattern;
- (void)addPatternsFromArray:(NSArray <RNNPattern *> *)patterns;
- (void)addPatternsFromFeatures:(NSArray <NSArray <NSNumber *> *> *)features targets:(NSArray <NSArray <NSNumber *> *> *)targets;

- (void)randomizeWeights;
- (void)uniformActiviation:(RNNNetActivations)activation;
- (void)uniformOptimization:(RNNOptimization *)optimization;
- (void)createHiddenLayerNetsForCount:(NSInteger)count;
- (void)createOutputLayerNetsForCount:(NSInteger)count;

- (void)trainingWithCompletion:(RNNCompletion)completion;
- (void)trainingWithBeforeUpdate:(RNNBeforeUpdate)beforeUpdate completion:(RNNCompletion)completion;
- (void)trainingWithIteration:(RNNIteration)iterationBlock completion:(RNNCompletion)completion;
- (void)predicateWithPatterns:(NSArray <RNNPattern *> *)patterns completion:(RNNPredication)completion;
- (void)predicateWithFeatures:(NSArray <NSNumber *> *)features completion:(RNNPredication)completion;

- (void)setTrainingCompletion:(RNNCompletion)trainingCompletion;
- (void)setTrainingIteraion:(RNNIteration)trainingIteraion;
- (void)setBeforeUpdate:(RNNBeforeUpdate)beforeUpdate;

@end
