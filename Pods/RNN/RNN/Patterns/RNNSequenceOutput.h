//
//  RNNOutputs.h
//  RNN
//
//  Created by kalvar_lin on 2017/7/5.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNSequenceOutput : NSObject

@property (nonatomic) NSInteger timestep;
@property (nonatomic) NSMutableArray <NSNumber *> *networkOutputs;

- (instancetype)initWithTimestep:(NSInteger)timestep networkOutputs:(NSArray <NSNumber *> *)networkOutputs;

@end
