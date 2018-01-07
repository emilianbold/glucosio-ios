//
//  RNNPartial.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNPartial : NSObject

- (double)tanh:(double)x slope:(double)slope;
- (double)sigmoid:(double)x slope:(double)slope;
- (double)rbf:(double)x sigma:(double)sigma;
- (double)sgn:(double)x;
- (double)reLU:(double)x;
- (double)leakyReLU:(double)x;
- (double)eLU:(double)x;

@end
