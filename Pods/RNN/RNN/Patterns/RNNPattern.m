//
//  RNNPattern.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/20.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNPattern.h"

@interface RNNPattern()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation RNNPattern (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key
{
    if( nil != object )
    {
        [self.coder encodeObject:object forKey:key];
    }
}

- (id)decodeForKey:(NSString *)key
{
    return [self.coder decodeObjectForKey:key];
}

@end

@implementation RNNPattern (Copy)

- (NSMutableArray <NSNumber *> *)copyArray:(NSArray <NSNumber *> *)array
{
    // [nil count] is 0
    return ([array count] > 0) ? [[NSMutableArray alloc] initWithArray:array copyItems:YES] : [[NSMutableArray alloc] init];
}

@end

@implementation RNNPattern

- (instancetype)initWithFeatures:(NSArray<NSNumber *> *)features targets:(NSArray<NSNumber *> *)targets
{
    self = [super init];
    if(self)
    {
        _features = [self copyArray:features];
        _targets  = [self copyArray:targets];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithFeatures:nil targets:nil];
}

- (void)addFeaturesFromArray:(NSArray<NSNumber *> *)features
{
    if([features count] > 0)
    {
        [_features addObjectsFromArray:[features copy]];
    }
}

- (void)addTargetsFromArray:(NSArray<NSNumber *> *)targets
{
    if([targets count] > 0)
    {
        [_targets addObjectsFromArray:[targets copy]];
    }
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    return [[RNNPattern alloc] initWithFeatures:_features targets:_targets];
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    _coder = aCoder;
    [self encodeObject:_features forKey:@"features"];
    [self encodeObject:_targets forKey:@"targets"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        _coder    = aDecoder;
        _features = [self decodeForKey:@"features"];
        _targets  = [self decodeForKey:@"targets"];
    }
    return self;
}

@end
