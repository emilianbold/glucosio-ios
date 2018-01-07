//
//  RNNTimestep.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/27.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNTimestep.h"

@implementation RNNTimestep

- (instancetype)init
{
    self = [super init];
    if(self)
    {
        _deltaBias             = 0.0f;
        _deltaWeights          = [[NSMutableArray alloc] init];
        _recurrentDeltaWeights = [[NSMutableArray alloc] init];
        _gradients             = [[NSMutableArray alloc] init];
        _recurrentGradients    = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)addDeltaWeight:(NSNumber *)deltaWeight
{
    if(nil != deltaWeight)
    {
        [_deltaWeights addObject:deltaWeight];
    }
}

- (void)addRecurrentDeltaWeight:(NSNumber *)recurrentWeight
{
    if(nil != recurrentWeight)
    {
        [_recurrentDeltaWeights addObject:recurrentWeight];
    }
}

- (void)addGradient:(NSNumber *)gradientWeight
{
    if(nil != gradientWeight)
    {
        [_gradients addObject:gradientWeight];
    }
}

- (void)addRecurrentGradient:(NSNumber *)gradientWeight
{
    if(nil != gradientWeight)
    {
        [_recurrentGradients addObject:gradientWeight];
    }
}

- (double)deltaWeightForIndex:(NSInteger)weightIndex
{
    return [[_deltaWeights objectAtIndex:weightIndex] doubleValue];
}

- (double)recurrentDeltaWeightForIndex:(NSInteger)recurrentIndex
{
    return [[_recurrentDeltaWeights objectAtIndex:recurrentIndex] doubleValue];
}

@end
