//
//  RNNTimestep.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/27.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNTimestep : NSObject

@property (nonatomic) double deltaBias;
@property (nonatomic) NSMutableArray <NSNumber *> *deltaWeights;
@property (nonatomic) NSMutableArray <NSNumber *> *recurrentDeltaWeights;
@property (nonatomic) NSMutableArray <NSNumber *> *gradients;
@property (nonatomic) NSMutableArray <NSNumber *> *recurrentGradients;

- (void)addDeltaWeight:(NSNumber *)deltaWeight;
- (void)addRecurrentDeltaWeight:(NSNumber *)recurrentWeight;
- (void)addGradient:(NSNumber *)gradientWeight;
- (void)addRecurrentGradient:(NSNumber *)gradientWeight;

- (double)deltaWeightForIndex:(NSInteger)weightIndex;
- (double)recurrentDeltaWeightForIndex:(NSInteger)recurrentIndex;

@end
