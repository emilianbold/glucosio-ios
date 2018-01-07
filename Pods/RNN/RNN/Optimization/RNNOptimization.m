//
//  RNNOptimization.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNOptimization.h"
#import "RNNNet.h"

@interface RNNOptimization ()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation RNNOptimization (NSCoding)

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

@implementation RNNOptimization

+ (instancetype)shared
{
    static dispatch_once_t pred;
    static RNNOptimization *object = nil;
    dispatch_once(&pred, ^{
        object = [[RNNOptimization alloc] init];
    });
    return object;
}

- (instancetype)init
{
    self = [super init];
    if(self)
    {
        _method                    = RNNOptimizationStandardSGD;
        _inertialRate              = 0.5f;
        _lastDeltaWeights          = [[NSMutableArray alloc] init];
        _lastRecurrentDeltaWeights = [[NSMutableArray alloc] init];
    }
    return self;
}

- (void)fillZeroToLastDeltaWeightsForCount:(NSInteger)count
{
    [_lastDeltaWeights removeAllObjects];
    [_lastRecurrentDeltaWeights removeAllObjects];
    
    for(NSInteger i=0; i<count; i++)
    {
        [_lastDeltaWeights addObject:@(0.0f)];
        [_lastRecurrentDeltaWeights addObject:@(0.0f)];
    }
}

- (void)recordDeltaWeights:(NSArray <NSNumber *> *)deltaWeights recurrentDeltaWeights:(NSArray <NSNumber *> *)recurrentWeights
{
    if([deltaWeights count] > 0)
    {
        [_lastDeltaWeights removeAllObjects];
        [_lastDeltaWeights addObjectsFromArray:deltaWeights];
    }
    
    if([recurrentWeights count] > 0)
    {
        [_lastRecurrentDeltaWeights removeAllObjects];
        [_lastRecurrentDeltaWeights addObjectsFromArray:recurrentWeights];
    }
}

/*
 * @ Parameters
 *   - weightIndex: Which one weight we wanna calculate its delta value.
 *   - net: Updating target net.
 *   - lastLayerOutput: The target net mapped with net output-value of last layer.
 */
- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(RNNNet *)net lastLayerOutput:(double)lastLayerOutput learningRate:(double)learningRate isRecurrent:(BOOL)isRecurrent
{
    // SGD: new w = old w + (- learning rate * - error value * f'(net) * x
    double deltaWeight = learningRate * net.deltaValue * lastLayerOutput;
    switch (_method) {
        case RNNOptimizationFixedInertia:
        {
            // 慣性項
            NSArray <NSNumber *> *deltas = isRecurrent ? _lastRecurrentDeltaWeights : _lastDeltaWeights;
            deltaWeight += (_inertialRate * [[deltas objectAtIndex:weightIndex] doubleValue]);
        }
            break;
        case RNNOptimizationStandardSGD:
        default:
            break;
    }
    
    return deltaWeight;
}

- (double)deltaWeightAtIndex:(NSInteger)weightIndex net:(RNNNet *)net lastLayerOutput:(double)lastLayerOutput learningRate:(double)learningRate
{
    return [self deltaWeightAtIndex:weightIndex net:net lastLayerOutput:lastLayerOutput learningRate:learningRate isRecurrent:NO];
}

#pragma mark - Getters
- (BOOL)needToFillZero
{
    return [_lastDeltaWeights count] == 0 || [_lastRecurrentDeltaWeights count] == 0;
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    RNNOptimization *optimization = [[RNNOptimization alloc] init];
    optimization.method                    = _method;
    optimization.inertialRate              = _inertialRate;
    optimization.lastDeltaWeights          = [[NSMutableArray alloc] initWithArray:_lastDeltaWeights copyItems:YES];
    optimization.lastRecurrentDeltaWeights = [[NSMutableArray alloc] initWithArray:_lastRecurrentDeltaWeights copyItems:YES];
    return optimization;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:@(_method) forKey:@"method"];
    [self encodeObject:@(_inertialRate) forKey:@"inertialRate"];
    [self encodeObject:_lastDeltaWeights forKey:@"lastDeltaWeights"];
    [self encodeObject:_lastRecurrentDeltaWeights forKey:@"lastRecurrentDeltaWeights"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder = aDecoder;
        _method                    = [[self decodeForKey:@"method"] integerValue];
        _inertialRate              = [[self decodeForKey:@"inertialRate"] doubleValue];
        _lastDeltaWeights          = [self decodeForKey:@"lastDeltaWeights"];
        _lastRecurrentDeltaWeights = [self decodeForKey:@"lastRecurrentDeltaWeights"];
        
    }
    return self;
}

@end
