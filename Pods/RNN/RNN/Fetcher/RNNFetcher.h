//
//  RNNFetcher.h
//  RNN
//
//  Created by kalvar_lin on 2017/6/28.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import <Foundation/Foundation.h>

@class RNN;

@interface RNNFetcher : NSObject

+ (instancetype)sharedFetcher;

- (void)save:(RNN *)object forKey:(NSString *)key;
- (void)removeForKey:(NSString *)key;
- (RNN *)objectForKey:(NSString *)key;

@end
