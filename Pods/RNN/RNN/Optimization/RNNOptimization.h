//
//  RNNOptimization.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, RNNOptimizationMethods)
{
    RNNOptimizationStandardSGD,       // Standard SGD.
    RNNOptimizationFixedInertia       // The fixed inertia.
};

@class RNNNet;

@interface RNNOptimization : NSObject <NSCopying, NSCoding>

@property (nonatomic) RNNOptimizationMethods method;
@property (nonatomic) double inertialRate;
@property (nonatomic) NSMutableArray <NSNumber *> *lastDeltaWeights;          // For delta weights.
@property (nonatomic) NSMutableArray <NSNumber *> *lastRecurrentDeltaWeights; // For recurrent weights
@property (nonatomic, readonly) BOOL needToFillZero; // If lastDeltaWeights or lastRecurrentDeltaWeights is empty, this BOOL will be YES.

+ (instancetype)shared;

- (void)fillZeroToLastDeltaWeightsForCount:(NSInteger)count;
- (void)recordDeltaWeights:(NSArray <NSNumber *> *)deltaWeights recurrentDeltaWeights:(NSArray <NSNumber *> *)recurrentWeights;
- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(RNNNet *)net lastLayerOutput:(double)lastLayerOutput learningRate:(double)learningRate isRecurrent:(BOOL)isRecurrent;
- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(RNNNet *)net lastLayerOutput:(double)lastLayerOutput learningRate:(double)learningRate;

@end

@interface RNNOptimization (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key;
- (id)decodeForKey:(NSString *)key;

@end
