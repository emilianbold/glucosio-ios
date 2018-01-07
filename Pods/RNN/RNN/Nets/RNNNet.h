//
//  RNNNet.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNOptimization.h"
#import "RNNNetOutputs.h"

typedef NS_ENUM(NSInteger, RNNNetActivations)
{
    RNNNetActivationSGN = 0,
    RNNNetActivationTanh,
    RNNNetActivationSigmoid,
    RNNNetActivationRBF,
    RNNNetActivationReLU,
    RNNNetActivationELU,
    RNNNetActivationLeakyReLU
};

@interface RNNNet : NSObject <NSCopying, NSCoding>

@property (nonatomic) NSMutableArray <NSNumber *> *weights;
@property (nonatomic) NSMutableArray <NSNumber *> *recurrentWeights;
@property (nonatomic) double bias;

@property (nonatomic, readonly) double outputValue;    // The last output value from outputValues.
@property (nonatomic, readonly) double outputPartial;  // The last output value partial from outputValues.
@property (nonatomic, readonly) double previousOutput; // The last moment output. e.g. b[t-1][h]
@property (nonatomic, readonly) double previousOutputPartial;
@property (nonatomic) double deltaValue;               // Current delta value to be next delta value.
@property (nonatomic) BOOL hasRecurrent;               // Has recurrent inputs ?

@property (nonatomic) RNNNetActivations netActivation;
@property (nonatomic) RNNOptimization *optimization;

- (void)addWeightsFromArray:(NSArray <NSNumber *> *)weights;
- (void)addRecurrentWeightsFromArray:(NSArray <NSNumber *> *)recurrentWeights;

- (double)weightForIndex:(NSInteger)weightIndex;
- (double)recurrentWeightForIndex:(NSInteger)recurrentIndex;

- (void)removeAllWeights;
- (void)renewWeights:(NSArray <NSNumber *> *)newWeights recurrentWeights:(NSArray <NSNumber *> *)newRecurrents;
- (void)randomizeWeightsAtCount:(NSInteger)randomCount max:(double)max min:(double)min;

- (void)checkToFillZero;
- (void)clear;

- (double)outputWithInputs:(NSArray <NSNumber *> *)inputs recurrentOutputs:(NSArray <NSNumber *> *)recurrents;
- (double)outputWithInputs:(NSArray <NSNumber *> *)inputs;

- (void)calculateDeltaWeightsWithLayerOutputs:(NSArray <NSNumber *> *)layerOutputs recurrentOutputs:(NSArray <NSNumber *> *)recurrentOutputs learningRate:(double)learningRate;
- (void)calculateDeltaWeightsWithHiddenOutputs:(NSArray <NSNumber *> *)hiddenOutputs learningRate:(double)learningRate;
- (void)renew;

@end

@interface RNNNet (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key;
- (id)decodeForKey:(NSString *)key;

@end
