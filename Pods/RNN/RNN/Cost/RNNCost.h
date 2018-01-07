//
//  RNNCost.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/22.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNCost : NSObject

@property (nonatomic, readonly) NSInteger patternsCount;
@property (nonatomic, readonly) NSInteger outputsCount;
@property (nonatomic, readonly) double mse;
@property (nonatomic, readonly) double rmse;
@property (nonatomic, readonly) double crossEntropy;

- (instancetype)init;

- (void)addOutputs:(NSArray <NSNumber *> *)outputs targets:(NSArray <NSNumber *> *)targets;
- (void)removeAllObjects;

@end
