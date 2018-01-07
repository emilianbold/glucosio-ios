//
//  RNNOutputs.m
//  RNN
//
//  Created by kalvar_lin on 2017/7/5.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNSequenceOutput.h"

@implementation RNNSequenceOutput (Arrays)

- (NSMutableArray <NSNumber *> *)copyArray:(NSArray <NSNumber *> *)array
{
    return ([array count] > 0) ? [[NSMutableArray alloc] initWithArray:array copyItems:YES] : [[NSMutableArray alloc] init];
}

@end

@implementation RNNSequenceOutput

- (instancetype)initWithTimestep:(NSInteger)timestep networkOutputs:(NSArray <NSNumber *> *)networkOutputs
{
    self = [super init];
    if(self)
    {
        _timestep       = timestep;
        _networkOutputs = [self copyArray:networkOutputs];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithTimestep:0 networkOutputs:nil];
}

@end
