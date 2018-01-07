//
//  RNNMath.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNMath.h"

@implementation RNNMath

+(instancetype)shared
{
    static dispatch_once_t pred;
    static RNNMath *object = nil;
    dispatch_once(&pred, ^{
        object = [[RNNMath alloc] init];
    });
    return object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

-(double)sumMatrix:(NSArray *)_parentMatrix anotherMatrix:(NSArray *)_childMatrix
{
    double _sum      = 0.0f;
    NSInteger _index = 0;
    for( NSNumber *_value in _parentMatrix )
    {
        _sum += [_value doubleValue] * [[_childMatrix objectAtIndex:_index] doubleValue];
        ++_index;
    }
    return _sum;
}

-(NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number
{
    NSMutableArray *_array = [NSMutableArray new];
    for( NSNumber *_value in _matrix )
    {
        double _newValue = _number * [_value doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
    }
    return _array;
}

-(NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix
{
    NSMutableArray *_array = [NSMutableArray new];
    NSInteger _index       = 0;
    for( NSNumber *_value in _matrix )
    {
        double _newValue = [_value doubleValue] + [[_anotherMatrix objectAtIndex:_index] doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
        ++_index;
    }
    return _array;
}

- (double)randomDoubleMax:(double)_maxValue min:(double)_minValue
{
    if( _maxValue == _minValue )
    {
        return ( _maxValue == 0.0f ) ? 0.0f : _maxValue;
    }
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

@end
