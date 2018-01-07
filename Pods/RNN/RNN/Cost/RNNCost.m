//
//  RNNCost.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/22.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNCost.h"

@interface RNNCost ()

@property (nonatomic, readonly) double costValue;
@property (nonatomic) NSMutableArray <NSArray <NSNumber *> *> *outputs;
@property (nonatomic) NSMutableArray <NSArray <NSNumber *> *> *targets;

@end

@implementation RNNCost (Checks)

- (BOOL)canCalculate
{
    return (self.patternsCount != 0 && self.outputsCount != 0);
}

@end

@implementation RNNCost

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _outputs = [[NSMutableArray alloc] init];
        _targets = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)addOutputs:(NSArray<NSNumber *> *)outputs targets:(NSArray<NSNumber *> *)targets
{
    [_outputs addObject:[outputs copy]];
    [_targets addObject:[targets copy]];
}

- (void)removeAllObjects
{
    [_outputs removeAllObjects];
    [_targets removeAllObjects];
}

#pragma mark - Getters
- (double)costValue
{
    __block NSArray <NSArray <NSNumber *> *> *targets = _targets;
    __block double costValue = 0.0f;
    // 計算每一個範本的 Cost Value
    [_outputs enumerateObjectsUsingBlock:^(NSArray<NSNumber *> * _Nonnull patternOutputs, NSUInteger idx, BOOL * _Nonnull stop) {
        __block NSArray <NSNumber *> *patternTargets = [targets objectAtIndex:idx];
        [patternOutputs enumerateObjectsUsingBlock:^(NSNumber * _Nonnull netOutput, NSUInteger idx, BOOL * _Nonnull stop) {
            double outputValue = [netOutput doubleValue];
            double targetValue = [[patternTargets objectAtIndex:idx] doubleValue];
            double outputError = targetValue - outputValue;
            costValue += (outputError * outputError);
        }];
    }];
    return costValue;
}

- (NSInteger)patternsCount
{
    return [_outputs count];
}

- (NSInteger)outputsCount
{
    return [[_outputs firstObject] count];
}

- (double)mse
{
    return [self canCalculate] ? self.costValue / (self.patternsCount * self.outputsCount) * 0.5f : NSNotFound;
}

- (double)rmse
{
    return [self canCalculate] ? sqrt(self.costValue / (self.patternsCount * self.outputsCount)) : NSNotFound;
}

- (double)crossEntropy
{
    __block NSArray <NSArray <NSNumber *> *> *targets = _targets;
    __block double iterationEntropy = 0.0f;
    __block NSInteger outputCount   = self.outputsCount;
    [_outputs enumerateObjectsUsingBlock:^(NSArray<NSNumber *> * _Nonnull patternOutputs, NSUInteger idx, BOOL * _Nonnull stop) {
        __block NSArray <NSNumber *> *patternTargets = [targets objectAtIndex:idx];
        __block double patternEntropy = 0.0f;
        [patternOutputs enumerateObjectsUsingBlock:^(NSNumber * _Nonnull netOutput, NSUInteger idx, BOOL * _Nonnull stop) {
            double outputValue = [netOutput doubleValue];
            double targetValue = [[patternTargets objectAtIndex:idx] doubleValue];
            double entropy     = (targetValue * log(outputValue)) + ((1.0f - targetValue) * log(1.0f - outputValue));
            patternEntropy    += entropy;
        }];
        iterationEntropy += -(patternEntropy / outputCount);
    }];
    return (iterationEntropy / self.patternsCount);
}

@end
