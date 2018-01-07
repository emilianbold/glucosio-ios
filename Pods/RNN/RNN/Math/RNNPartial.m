//
//  RNNPartial.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNPartial.h"

@implementation RNNPartial

- (double)tanh:(double)x slope:(double)slope
{
    return (1.0f - ( x * x )) * (slope / 2.0f);
}

- (double)sigmoid:(double)x slope:(double)slope
{
    return slope * x * ( 1.0f - x );
}

- (double)rbf:(double)x sigma:(double)sigma
{
    return -((2.0f * x) / (2.0f * sigma * sigma)) * exp(-x / (2.0f * sigma * sigma));
}
- (double)sgn:(double)x
{
    return (x >= 0.0f) ? 1.0 : -1.0;
}

- (double)reLU:(double)x
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}

- (double)leakyReLU:(double)x
{
    return (x > 0.0f) ? 1.0f : 0.01f;
}

- (double)eLU:(double)x
{
    return (x >= 0.0f) ? x : 0.01 * exp(x);
}

@end
