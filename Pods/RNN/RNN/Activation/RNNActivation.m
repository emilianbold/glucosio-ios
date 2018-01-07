//
//  RNNActivations.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNActivation.h"

@implementation RNNActivation

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _partial = [[RNNPartial alloc] init];
    }
    return self;
}

- (double)tanh:(double)x slope:(double)slope
{
    return (2.0f / (1.0f + exp(-slope * x))) - 1.0f;
}

- (double)sigmoid:(double)x slope:(double)slope
{
    return (1.0f / (1.0f + exp(-slope * x)));
}

- (double)sgn:(double)x
{
    return (x >= 0.0f) ? 1.0f : -1.0f;
}

- (double)rbf:(double)x sigma:(double)sigma
{
    return exp(-x / (2.0f * sigma * sigma));
}

- (double)reLU:(double)x
{
    return MAX(0.0, x);
}

- (double)leakyReLU:(double)x
{
    return MAX(0.01f * x, x);
}

- (double)eLU:(double)x
{
    return (x >= 0.0f) ? x : 0.01 * (exp(x) - 1.0f);
}

@end
