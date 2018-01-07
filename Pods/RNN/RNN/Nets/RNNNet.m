//
//  RNNNet.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNNNet.h"
#import "RNNMath.h"
#import "RNNActivation.h"
#import "RNNTimestep.h"

@interface RNNNet ()

@property (nonatomic) RNNActivation *activation;
@property (nonatomic) RNNMath *math;
@property (nonatomic) RNNNetOutputs *output;
@property (nonatomic, weak) NSCoder *coder;
@property (nonatomic) NSMutableArray <RNNTimestep *> *timesteps;

@end

@implementation RNNNet (NSCoding)

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

@implementation RNNNet (Bias)

- (void)renewBias
{
    __block double sumChanges = 0.0f;
    [self.timesteps enumerateObjectsUsingBlock:^(RNNTimestep * _Nonnull timestep, NSUInteger idx, BOOL * _Nonnull stop) {
        sumChanges += timestep.deltaBias;
    }];
    
    // new b(j) = old b(j) + [-L * -delta(j)]
    self.bias += sumChanges;
}

@end

@implementation RNNNet (Activations)

- (double)partialOfNet:(double)value
{
    RNNPartial *partial = self.activation.partial;
    double partialValue = value;
    switch (self.netActivation)
    {
        case RNNNetActivationTanh:
            partialValue = [partial tanh:value slope:1.0f];
            break;
        case RNNNetActivationSigmoid:
            partialValue = [partial sigmoid:value slope:1.0f];
            break;
        case RNNNetActivationRBF:
            partialValue = [partial rbf:value sigma:1.0f];
            break;
        case RNNNetActivationSGN:
            partialValue = [partial sgn:value];
            break;
        case RNNNetActivationReLU:
            partialValue = [partial reLU:value];
            break;
        case RNNNetActivationELU:
            partialValue = [partial eLU:value];
            break;
        case RNNNetActivationLeakyReLU:
            partialValue = [partial leakyReLU:value];
            break;
        default:
            break;
    }
    return partialValue;
}

- (double)activate:(double)x
{
    double y = 0.0f;
    switch (self.netActivation)
    {
        case RNNNetActivationTanh:
            y = [self.activation tanh:x slope:1.0f];
            break;
        case RNNNetActivationSigmoid:
            y = [self.activation sigmoid:x slope:1.0f];
            break;
        case RNNNetActivationRBF:
            y = [self.activation rbf:x sigma:1.0f];
            break;
        case RNNNetActivationSGN:
            y = [self.activation sgn:x];
            break;
        case RNNNetActivationReLU:
            y = [self.activation reLU:x];
            break;
        case RNNNetActivationELU:
            y = [self.activation eLU:x];
            break;
        case RNNNetActivationLeakyReLU:
            y = [self.activation leakyReLU:x];
            break;
        default:
            break;
    }
    return y;
}

// 判斷活化函式是否為線性
- (BOOL)isLinear
{
    BOOL isLinearFunction = YES;
    switch (self.netActivation)
    {
        case RNNNetActivationTanh:
        case RNNNetActivationSigmoid:
        case RNNNetActivationRBF:
            isLinearFunction = NO;
            break;
        default:
            break;
    }
    return isLinearFunction;
}

@end

@implementation RNNNet

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _weights          = [[NSMutableArray alloc] init];
        _recurrentWeights = [[NSMutableArray alloc] init];
        
        _output           = [[RNNNetOutputs alloc] init];
        
        _bias             = 0.0f;
        _deltaValue       = 0.0f;
        _hasRecurrent     = NO;
        
        _netActivation    = RNNNetActivationSigmoid;
        _activation       = [[RNNActivation alloc] init];
        _math             = [[RNNMath alloc] init];
        
        _timesteps        = [[NSMutableArray alloc] init];
        
        _optimization              = [[RNNOptimization alloc] init];
        _optimization.method       = RNNOptimizationFixedInertia;
        _optimization.inertialRate = 0.5f;
        
    }
    return self;
}

- (void)addWeightsFromArray:(NSArray <NSNumber *> *)weights
{
    if( [weights count] > 0 )
    {
        [_weights addObjectsFromArray:weights];
    }
}

- (void)addRecurrentWeightsFromArray:(NSArray <NSNumber *> *)recurrentWeights
{
    if([recurrentWeights count] > 0)
    {
        [_recurrentWeights addObjectsFromArray:recurrentWeights];
    }
}

- (double)weightForIndex:(NSInteger)weightIndex
{
    return [[_weights objectAtIndex:weightIndex] doubleValue];
}

- (double)recurrentWeightForIndex:(NSInteger)recurrentIndex
{
    return [[_recurrentWeights objectAtIndex:recurrentIndex] doubleValue];
}

- (void)removeAllWeights
{
    [_weights removeAllObjects];
    [_recurrentWeights removeAllObjects];
    _bias = 0.0f;
}

- (void)renewWeights:(NSArray <NSNumber *> *)newWeights recurrentWeights:(NSArray <NSNumber *> *)newRecurrents
{
    if(nil != newWeights && nil != newRecurrents)
    {
        [self removeAllWeights];
        [self addWeightsFromArray:newWeights];
        [self addRecurrentWeightsFromArray:newRecurrents];
    }
}

// To check out all arraies do they need to fill up zero objects first to avoid out of bound crash ?
// If yes it need, this function is going to fill the zero objects in arries.
- (void)checkToFillZero
{
    if(_optimization.needToFillZero)
    {
        [_optimization fillZeroToLastDeltaWeightsForCount:[_weights count]];
    }
}

// To be initial preparing status.
- (void)clear
{
    [self checkToFillZero];
    [_output refresh];
}

// Output of net.
- (double)outputWithInputs:(NSArray<NSNumber *> *)inputs recurrentOutputs:(NSArray<NSNumber *> *)recurrents
{
    double summedSignal = [_math sumMatrix:inputs anotherMatrix:_weights] + _bias;
    if(nil != recurrents)
    {
        summedSignal += [_math sumMatrix:recurrents anotherMatrix:_recurrentWeights];
    }
    double outputValue  = [self activate:summedSignal];
    [_output.sumInputs addObject:@(summedSignal)];
    [_output.outputValues addObject:@(outputValue)];
    return outputValue;
}

- (double)outputWithInputs:(NSArray<NSNumber *> *)inputs
{
    return [self outputWithInputs:inputs recurrentOutputs:nil];
}

#pragma mark - Randoms
// Randomizing for weights and recurrent weights at the same time.
- (void)randomizeWeightsAtCount:(NSInteger)randomCount max:(double)max min:(double)min
{
    [self removeAllWeights];
    for( NSInteger i=0; i<randomCount; i++ )
    {
        [_weights addObject:@([_math randomDoubleMax:max min:min])];
        if(_hasRecurrent)
        {
            [_recurrentWeights addObject:@([_math randomDoubleMax:max min:min])];
        }
    }
    
    _bias = [_math randomDoubleMax:max min:min];
}

#pragma mark - Updating
// For hidden layer nets to calculate their delta weights with recurrent layer.
- (void)calculateDeltaWeightsWithLayerOutputs:(NSArray <NSNumber *> *)layerOutputs recurrentOutputs:(NSArray <NSNumber *> *)recurrentOutputs learningRate:(double)learningRate
{
    // 利用 RNNTimestep 來當每一個 BP timestep 算權重修正值時的記錄容器
    __block RNNTimestep *timestep = [[RNNTimestep alloc] init];
    
    // For delta bias.
    timestep.deltaBias = learningRate * _deltaValue;
    
    // For delta weights.
    __weak typeof(self) weakSelf = self;
    [_weights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull weight, NSUInteger weightIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        
        // To calculate gradient of weight and delta of weight.
        double lastLayerOutput = [[layerOutputs objectAtIndex:weightIndex] doubleValue];
        double gradientWeight  = strongSelf.deltaValue * lastLayerOutput; // aE/aw(ij)
        [timestep addGradient:@(gradientWeight)];
        
        double deltaWeight     = [strongSelf.optimization deltaWeightAtIndex:weightIndex net:strongSelf lastLayerOutput:lastLayerOutput learningRate:learningRate];
        [timestep addDeltaWeight:@(deltaWeight)];
    }];
    
    [_recurrentWeights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull recurrentWeight, NSUInteger recurrentIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        
        double lastRecurrentOutput = [recurrentWeight doubleValue];
        double recurrentGradient   = strongSelf.deltaValue * lastRecurrentOutput;
        [timestep addRecurrentGradient:@(recurrentGradient)];
        
        double recurrentDeltaWeight = [strongSelf.optimization deltaWeightAtIndex:recurrentIndex net:strongSelf lastLayerOutput:lastRecurrentOutput learningRate:learningRate isRecurrent:YES];
        [timestep addRecurrentDeltaWeight:@(recurrentDeltaWeight)];
    }];
    
    [_timesteps addObject:timestep];
}

// For output layer nets to calculate their delta weights without recurrent layer.
- (void)calculateDeltaWeightsWithHiddenOutputs:(NSArray <NSNumber *> *)hiddenOutputs learningRate:(double)learningRate
{
    [self calculateDeltaWeightsWithLayerOutputs:hiddenOutputs recurrentOutputs:nil learningRate:learningRate];
}

// Renew weights and bias.
- (void)renew
{
    // 累加每個 Timestep 裡相同 Index 的 Delta Weight
    __block NSMutableArray <NSNumber *> *deltaWeights          = [[NSMutableArray alloc] init];
    __block NSMutableArray <NSNumber *> *recurrentDeltaWeights = [[NSMutableArray alloc] init];
    __block NSMutableArray <NSNumber *> *newWeights            = [[NSMutableArray alloc] init];
    __block NSMutableArray <NSNumber *> *newRecurrentWeights   = [[NSMutableArray alloc] init];
    
    __weak typeof(self) weakSelf = self;
    // weights and recurrent weights their dimensons are same.
    [_weights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull weight, NSUInteger weightIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        
        __block double sumDeltaChanges = 0.0f; // For normal weight.
        [strongSelf.timesteps enumerateObjectsUsingBlock:^(RNNTimestep * _Nonnull timestep, NSUInteger stepIndex, BOOL * _Nonnull stop) {
            sumDeltaChanges += [timestep deltaWeightForIndex:weightIndex];
        }];
        
        double newWeight = [weight doubleValue] + sumDeltaChanges;
        [newWeights addObject:@(newWeight)];
        
        // Recording the delta-weight for per single weight to be last delta weights.
        [deltaWeights addObject:@(sumDeltaChanges)];
        
    }];
    
    [_recurrentWeights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull recurrentWeight, NSUInteger recurrentIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        
        __block double sumRecurrentChanges = 0.0f; // for recurrent weight.
        [strongSelf.timesteps enumerateObjectsUsingBlock:^(RNNTimestep * _Nonnull timestep, NSUInteger stepIndex, BOOL * _Nonnull stop) {
            sumRecurrentChanges += [timestep recurrentDeltaWeightForIndex:recurrentIndex];
        }];
        
        double newRecurrentWeight = [[strongSelf.recurrentWeights objectAtIndex:recurrentIndex] doubleValue] + sumRecurrentChanges;
        [newRecurrentWeights addObject:@(newRecurrentWeight)];
        
        [recurrentDeltaWeights addObject:@(sumRecurrentChanges)];
    }];
    
    [self renewWeights:newWeights recurrentWeights:newRecurrentWeights];
    [self renewBias];
    
    [_timesteps removeAllObjects];
    [_optimization recordDeltaWeights:deltaWeights recurrentDeltaWeights:recurrentDeltaWeights];
    _deltaValue = 0.0f; // 一定要歸零，因為走 BPTT 的原故
}

#pragma mark - Getter
- (double)outputValue
{
    return _output.lastOutputValue;
}

- (double)outputPartial
{
    return [self partialOfNet:([self isLinear] ? _output.lastSumInput : self.outputValue)];
}

- (double)previousOutput
{
    return [_output previousOutput];
}

- (double)previousOutputPartial
{
    return [self partialOfNet:([self isLinear] ? [_output previousSumInput] : self.previousOutput)];
}

#pragma mark - NSCopying
- (instancetype)copyWithZone:(NSZone *)zone
{
    RNNNet *net          = [[RNNNet alloc] init];
    net.weights          = [[NSMutableArray alloc] initWithArray:_weights copyItems:YES];
    net.recurrentWeights = [[NSMutableArray alloc] initWithArray:_recurrentWeights copyItems:YES];
    net.output           = [_output copy];
    net.bias             = _bias;
    net.netActivation    = _netActivation;
    net.deltaValue       = _deltaValue;
    net.hasRecurrent     = _hasRecurrent;
    net.optimization     = [_optimization copy];
    return net;
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:_weights forKey:@"weights"];
    [self encodeObject:_recurrentWeights forKey:@"recurrentWeights"];
    [self encodeObject:_output forKey:@"output"];
    [self encodeObject:@(_deltaValue) forKey:@"deltaValue"];
    [self encodeObject:@(_hasRecurrent) forKey:@"hasRecurrent"];
    [self encodeObject:@(_bias) forKey:@"bias"];
    [self encodeObject:@(_netActivation) forKey:@"netActivation"];
    [self encodeObject:_optimization forKey:@"optimization"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder        = aDecoder;
        _weights          = [self decodeForKey:@"weights"];
        _recurrentWeights = [self decodeForKey:@"recurrentWeights"];
        _output           = [self decodeForKey:@"output"];
        _bias             = [[self decodeForKey:@"bias"] doubleValue];
        _netActivation    = [[self decodeForKey:@"netActivation"] integerValue];
        _hasRecurrent     = [[self decodeForKey:@"hasRecurrent"] boolValue];
        _optimization     = [self decodeForKey:@"optimization"];
        
        // Initializes objects without saved.
        _activation       = [[RNNActivation alloc] init];
        _math             = [[RNNMath alloc] init];
        _deltaValue       = 0.0f;
        _timesteps        = [[NSMutableArray alloc] init];
    }
    return self;
}

@end
