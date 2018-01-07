//
//  RNNMath.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNMath : NSObject

- (double)sumMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix;
- (NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number;
- (NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix;
- (double)randomDoubleMax:(double)_maxValue min:(double)_minValue;

@end
