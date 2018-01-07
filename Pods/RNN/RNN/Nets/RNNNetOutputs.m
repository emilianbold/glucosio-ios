//
//  RNNNetOutputs.m
//  RNN
//
//  Created by kalvar_lin on 2017/7/4.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNNetOutputs.h"

@interface RNNNetOutputs ()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation RNNNetOutputs (NSCoding)

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

@implementation RNNNetOutputs (Outputs)

- (void)initializeOutputs
{
    [self.sumInputs addObject:@(0.0f)];
    [self.outputValues addObject:@(0.0f)];
}

@end

@implementation RNNNetOutputs

- (instancetype)init
{
    self = [super init];
    if(self)
    {
        _sumInputs    = [[NSMutableArray alloc] init];
        _outputValues = [[NSMutableArray alloc] init];
        [self initializeOutputs];
    }
    return self;
}

- (void)refresh
{
    [_sumInputs removeAllObjects];
    [_outputValues removeAllObjects];
    [self initializeOutputs];
}

- (void)removeLastOutput
{
    [_sumInputs removeLastObject];
    [_outputValues removeLastObject];
}

#pragma mark - Getters
- (double)lastSumInput
{
    return [[_sumInputs lastObject] doubleValue];
}

- (double)lastOutputValue
{
    return [[_outputValues lastObject] doubleValue];
}

// Last moment output value.
- (double)previousOutput
{
    NSInteger index = self.previousIndex;
    return (index >= 0) ? [[_outputValues objectAtIndex:index] doubleValue] : 0.0f;
}

- (double)previousSumInput
{
    NSInteger index = self.previousIndex;
    return (index >= 0) ? [[_sumInputs objectAtIndex:index] doubleValue] : 0.0f;
}

- (NSInteger)previousIndex
{
    // 當前的 Index 的上一刻 Index，故 -2
    return self.outputsCount - 2;
}

- (NSInteger)outputsCount
{
    return [_outputValues count];
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    RNNNetOutputs *output = [[RNNNetOutputs alloc] init];
    output.sumInputs      = [[NSMutableArray alloc] initWithArray:_sumInputs copyItems:YES];
    output.outputValues   = [[NSMutableArray alloc] initWithArray:_outputValues copyItems:YES];
    return output;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:_sumInputs forKey:@"_sumInputs"];
    [self encodeObject:_outputValues forKey:@"_outputValues"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder    = aDecoder;
        _sumInputs    = [self decodeForKey:@"_sumInputs"];
        _outputValues = [self decodeForKey:@"_outputValues"];
    }
    return self;
}

@end
