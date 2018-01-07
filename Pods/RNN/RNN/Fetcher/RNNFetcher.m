//
//  RNNFetcher.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/28.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNFetcher.h"
#import "RNN.h"

@implementation RNNFetcher

+ (instancetype)sharedFetcher
{
    static dispatch_once_t pred;
    static RNNFetcher *object = nil;
    dispatch_once(&pred, ^{
        object = [[RNNFetcher alloc] init];
    });
    return object;
}

- (void)save:(RNN *)object forKey:(NSString *)key
{
    if(object && key)
    {
        [[NSUserDefaults standardUserDefaults] setObject:[NSKeyedArchiver archivedDataWithRootObject:object] forKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)removeForKey:(NSString *)key
{
    if(key)
    {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (RNN *)objectForKey:(NSString *)key
{
    if(key)
    {
        NSData *objectData = [[NSUserDefaults standardUserDefaults] valueForKey:key];
        return objectData ? [NSKeyedUnarchiver unarchiveObjectWithData:objectData] : nil;
    }
    return nil;
}

@end
