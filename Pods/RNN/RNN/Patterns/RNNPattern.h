//
//  RNNPattern.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/20.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@interface RNNPattern : NSObject <NSCopying, NSCoding>

@property (nonatomic) NSMutableArray <NSNumber *> *features;
@property (nonatomic) NSMutableArray <NSNumber *> *targets;

- (instancetype)initWithFeatures:(NSArray <NSNumber *> *)features targets:(NSArray <NSNumber *> *)targets;
- (void)addFeaturesFromArray:(NSArray <NSNumber *> *)features;
- (void)addTargetsFromArray:(NSArray <NSNumber *> *)targets;

@end
