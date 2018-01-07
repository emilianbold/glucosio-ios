//
//  RNNActivations.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNPartial.h"

@interface RNNActivation : NSObject

@property (nonatomic, readonly) RNNPartial *partial;

- (double)tanh:(double)x slope:(double)slope;
- (double)sigmoid:(double)x slope:(double)slope;
- (double)sgn:(double)x;
- (double)rbf:(double)x sigma:(double)sigma;
- (double)reLU:(double)x;
- (double)leakyReLU:(double)x;
- (double)eLU:(double)x;

@end
