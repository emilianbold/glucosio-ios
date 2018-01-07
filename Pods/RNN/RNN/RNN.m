//
//  RNN.m
//  RNN
//
//  Created by kalvar_lin on 2017/6/16.
//  Copyright © 2017年 Kalvar Lin, the Knowledge Intelligence (K.I.). All rights reserved.
//

#import "RNN.h"

@interface RNN ()

@property (nonatomic) RNNCost *cost;
@property (nonatomic, weak) NSCoder *coder;

@end

@implementation RNN (NSCoding)

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

@implementation RNN (Setups)

- (void)setup
{
    // Filling up all settings of default.
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull net, NSUInteger idx, BOOL * _Nonnull stop) {
        [net checkToFillZero];
    }];
    
    [self.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull net, NSUInteger idx, BOOL * _Nonnull stop) {
        [net checkToFillZero];
    }];
}

- (void)createNetsForCount:(NSInteger)count toLayer:(NSMutableArray <RNNNet *> *)layer hasRecurrent:(BOOL)hasRecurrent
{
    for(NSInteger i=0; i<count; i++)
    {
        RNNNet *net      = [[RNNNet alloc] init];
        net.hasRecurrent = hasRecurrent;
        [layer addObject:net];
    }
}

@end

@implementation RNN (Training)

- (void)bpttUpdate
{
    // Before updating the weights.
    if( nil != self.beforeUpdate )
    {
        self.beforeUpdate(self.iteration, self);
    }
    
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [hiddenNet renew];
    }];
    
    [self.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [outputNet renew];
    }];
}

- (NSArray <NSNumber *> *)forwardWithFeatures:(NSArray <NSNumber *> *)features
{
    // Fetching recurrent layer nets.
    __block NSMutableArray <NSNumber *> *recurrentOutputs = [[NSMutableArray alloc] init];
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [recurrentOutputs addObject:@(hiddenNet.outputValue)]; // 此時的 OutputValue 才是 Recurrent Output (Previous Output)
    }];
    
    // Forward
    __block NSMutableArray <NSNumber *> *hiddenOutputs = [[NSMutableArray alloc] init];
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        double netOutput = [hiddenNet outputWithInputs:features recurrentOutputs:recurrentOutputs];
        [hiddenOutputs addObject:@(netOutput)];
    }];
    
    __block NSMutableArray <NSNumber *> *networkOutputs = [[NSMutableArray alloc] init];
    [self.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        double netOutput = [outputNet outputWithInputs:hiddenOutputs];
        [networkOutputs addObject:@(netOutput)];
    }];
    
    return networkOutputs;
}

- (void)trainingWithPattern:(RNNPattern *)pattern
{
    __block NSArray <NSNumber *> *networkOutputs = [self forwardWithFeatures:pattern.features];
    
    [self.cost addOutputs:networkOutputs targets:pattern.targets];
    
    // Backward
    __weak typeof(self) weakSelf = self;
    // To calculate the deltas of output-layer.
    [self.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger outputIndex, BOOL * _Nonnull stop) {
        // Calculating deltas of output nets.
        double targetValue   = [[pattern.targets objectAtIndex:outputIndex] doubleValue];
        double outputValue   = [[networkOutputs objectAtIndex:outputIndex] doubleValue];
        double outputError   = targetValue - outputValue;
        outputNet.deltaValue = (outputError * outputNet.outputPartial);
    }];
    
    // To calculate the deltas of output-layer to hidden-layer, and record the outputs of nets to an array.
    __block NSMutableArray <NSNumber *> *hiddenOutputs    = [[NSMutableArray alloc] init];
    __block NSMutableArray <NSNumber *> *recurrentOutputs = [[NSMutableArray alloc] init];
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger hiddenIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        
        // Output-layer to hidden-layer.
        __block double sumDelta = 0.0f;
        [strongSelf.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger outputIndex, BOOL * _Nonnull stop) {
            // delta[t][k]
            sumDelta += outputNet.deltaValue * [outputNet weightForIndex:hiddenIndex];
        }];
        
        // Recurrent-layer to hidden-layer.
        __block double sumRecurrentDelta = 0.0f;
        [strongSelf.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull recurrentNet, NSUInteger recurrentIndex, BOOL * _Nonnull stop) {
            // delta[t+1][h] the recurrent delta.
            sumRecurrentDelta += recurrentNet.deltaValue * [recurrentNet recurrentWeightForIndex:hiddenIndex];
        }];
        
        // Hidden nets delta: (SUM(output-net-delta[t][k] * w(jk)) + SUM(recurrent-net-delta[t+1][h] * w(h'h))) * f'(hidden-net-output)
        hiddenNet.deltaValue = (sumDelta + sumRecurrentDelta) * hiddenNet.outputPartial;
        
        // To record the net output for backpropagation updating.
        [hiddenOutputs addObject:@(hiddenNet.outputValue)];
        [recurrentOutputs addObject:@(hiddenNet.previousOutput)];
    }];
    
    __block double learningRate = self.learningRate;
    __block NSArray <NSNumber *> *inputs = pattern.features;
    
    // To update weights of net-by-net between output-layer, hidden-layer and input-layer.
    [self.outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger outputIndex, BOOL * _Nonnull stop) {
        [outputNet calculateDeltaWeightsWithHiddenOutputs:hiddenOutputs learningRate:learningRate];
    }];
    
    // To fetch the outputs of input-layer for updating weights of hidden-layer.
    [self.hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger hiddenIndex, BOOL * _Nonnull stop) {
        [hiddenNet calculateDeltaWeightsWithLayerOutputs:inputs recurrentOutputs:recurrentOutputs learningRate:learningRate];
    }];
}

- (void)startTraining
{
    [self.cost removeAllObjects];
    self.iteration += 1;
    __block NSInteger total = [self.inputLayer count];
    __weak typeof(self) weakSelf = self;
    [self.inputLayer enumerateObjectsUsingBlock:^(RNNPattern * _Nonnull pattern, NSUInteger patternIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        [strongSelf trainingWithPattern:pattern];
        NSInteger currentSize = patternIndex + 1;
        // 是最後一筆 or Full-BPTT
        if(currentSize == total)
        {
            [strongSelf bpttUpdate];
        }
        else
        {
            // 先寫 Full-BPTT，之後再重構寫成 Truncated-BPTT (目前這種寫法還不能運作 Truncated-BPTT)
            if(strongSelf.timestepSize > kRNNFullBPTT && currentSize % strongSelf.timestepSize == 0)
            {
                [strongSelf bpttUpdate];
            }
        }
    }];
    
    // One iteration done then doing next adjust conditions
    if( self.iteration >= self.maxIteration || self.cost.mse <= self.convergenceError )
    {
        if( nil != self.trainingCompletion )
        {
            self.trainingCompletion(self.iteration, self);
        }
    }
    else
    {
        if( nil != self.trainingIteraion )
        {
            self.trainingIteraion(self.iteration, self);
        }
        [self startTraining];
    }
}

@end

@implementation RNN

+ (instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static RNN *object = nil;
    dispatch_once(&pred, ^{
        object = [[RNN alloc] init];
    });
    return object;
}

- (instancetype)init
{
    self = [super init];
    if(self)
    {
        _inputLayer         = [[NSMutableArray alloc] init];
        _hiddenLayer        = [[NSMutableArray alloc] init];
        _outputLayer        = [[NSMutableArray alloc] init];
        
        _learningRate       = 0.8f;
        
        _randomMax          = 0.5f;
        _randomMin          = -0.5f;
        
        _convergenceError   = 0.001f;
        _maxIteration       = 100;
        _iteration          = 0;
        _timestepSize       = kRNNFullBPTT; // Only support Full-BPTT recently.
        
        _trainingCompletion = nil;
        _trainingIteraion   = nil;
        _beforeUpdate       = nil;
        
        _cost               = [[RNNCost alloc] init];
    }
    return self;
}

- (void)addHiddenNet:(RNNNet *)net
{
    if(nil != net)
    {
        [_hiddenLayer addObject:net];
    }
}

- (void)addOutputNet:(RNNNet *)net
{
    if(nil != net)
    {
        [_outputLayer addObject:net];
    }
}

- (void)addPatternWithFeatures:(NSArray<NSNumber *> *)features targets:(NSArray<NSNumber *> *)targets
{
    RNNPattern *pattern = [[RNNPattern alloc] initWithFeatures:features targets:targets];
    [_inputLayer addObject:pattern];
}

- (void)addPattern:(RNNPattern *)pattern
{
    if(nil != pattern)
    {
        [_inputLayer addObject:pattern];
    }
}

- (void)addPatternsFromArray:(NSArray<RNNPattern *> *)patterns
{
    if([patterns count] > 0)
    {
        [_inputLayer addObjectsFromArray:patterns];
    }
}

- (void)addPatternsFromFeatures:(NSArray<NSArray<NSNumber *> *> *)features targets:(NSArray<NSArray<NSNumber *> *> *)targets
{
    __block BOOL hasTargets = (nil != targets);
    __weak typeof(self) weakSelf = self;
    [features enumerateObjectsUsingBlock:^(id  _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        NSArray <NSNumber *> *goals = hasTargets ? [targets objectAtIndex:idx] : nil;
        RNNPattern *pattern = [[RNNPattern alloc] initWithFeatures:obj targets:goals];
        [strongSelf addPattern:pattern];
    }];
}

- (void)randomizeWeights
{
    __block double max = _randomMax;
    __block double min = _randomMin;
    __block NSInteger inputCount  = [[_inputLayer firstObject].features count];
    __block NSInteger hiddenCount = [_hiddenLayer count];
    
    [_hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [hiddenNet randomizeWeightsAtCount:inputCount max:max min:min];
    }];
    
    [_outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [outputNet randomizeWeightsAtCount:hiddenCount max:max min:min];
    }];
}

- (void)uniformActiviation:(RNNNetActivations)activation
{
    _activation = activation;
    if(_activation != RNNNetActivationSigmoid)
    {
        __block RNNNetActivations netActiviation = _activation;
        [_hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
            hiddenNet.netActivation = netActiviation;
        }];
        
        [_outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
            outputNet.netActivation = netActiviation;
        }];
    }
}

- (void)uniformOptimization:(RNNOptimization *)optimization
{
    [_hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        hiddenNet.optimization = [optimization copy];
    }];
    
    [_outputLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull outputNet, NSUInteger idx, BOOL * _Nonnull stop) {
        outputNet.optimization = [optimization copy];
    }];
}

- (void)createHiddenLayerNetsForCount:(NSInteger)count
{
    [self createNetsForCount:count toLayer:_hiddenLayer hasRecurrent:YES];
}

- (void)createOutputLayerNetsForCount:(NSInteger)count
{
    [self createNetsForCount:count toLayer:_outputLayer hasRecurrent:NO];
}

- (void)trainingWithCompletion:(RNNCompletion)completion
{
    _trainingCompletion = completion;
    [self setup];
    [self startTraining];
}

- (void)trainingWithBeforeUpdate:(RNNBeforeUpdate)beforeUpdate completion:(RNNCompletion)completion
{
    _beforeUpdate = beforeUpdate;
    [self trainingWithCompletion:completion];
}

- (void)trainingWithIteration:(RNNIteration)iterationBlock completion:(RNNCompletion)completion
{
    _trainingIteraion = iterationBlock;
    [self trainingWithCompletion:completion];
}

- (void)predicateWithPatterns:(NSArray <RNNPattern *> *)patterns completion:(RNNPredication)completion
{
    // Let all hidden nets to be initial preparing status.
    [_hiddenLayer enumerateObjectsUsingBlock:^(RNNNet * _Nonnull hiddenNet, NSUInteger idx, BOOL * _Nonnull stop) {
        [hiddenNet clear];
    }];
    
    __block NSMutableArray <RNNSequenceOutput *> *outputs = [[NSMutableArray alloc] init];
    __weak typeof(self) weakSelf = self;
    [patterns enumerateObjectsUsingBlock:^(RNNPattern * _Nonnull pattern, NSUInteger patternIndex, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        NSArray <NSNumber *> *networkOutputs = [strongSelf forwardWithFeatures:pattern.features];
        RNNSequenceOutput *output            = [[RNNSequenceOutput alloc] initWithTimestep:(patternIndex + 1) networkOutputs:networkOutputs];
        [outputs addObject:output];
    }];
    
    if(completion)
    {
        completion(outputs);
    }
}

- (void)predicateWithFeatures:(NSArray <NSNumber *> *)features completion:(RNNPredication)completion
{
    RNNPattern *singlePattern = [[RNNPattern alloc] initWithFeatures:features targets:nil];
    [self predicateWithPatterns:@[singlePattern] completion:completion];
}

#pragma mark - Getters
- (RNNCost *)costFunction
{
    return _cost;
}

- (RNNFetcher *)fetcher
{
    return [RNNFetcher sharedFetcher];
}

#pragma mark - NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    _coder = aCoder;
    [self encodeObject:_hiddenLayer forKey:@"hiddenLayer"];
    [self encodeObject:_outputLayer forKey:@"outputLayer"];
    [self encodeObject:@(_activation) forKey:@"activation"];
    [self encodeObject:@(_learningRate) forKey:@"learningRate"];
    [self encodeObject:@(_randomMax) forKey:@"randomMax"];
    [self encodeObject:@(_randomMin) forKey:@"randomMin"];
    [self encodeObject:@(_convergenceError) forKey:@"convergenceError"];
    [self encodeObject:@(_maxIteration) forKey:@"maxIteration"];
    [self encodeObject:@(_timestepSize) forKey:@"batchSize"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        _coder            = aDecoder;
        _inputLayer       = [[NSMutableArray alloc] init];
        _hiddenLayer      = [self decodeForKey:@"hiddenLayer"];
        _outputLayer      = [self decodeForKey:@"outputLayer"];
        _activation       = [[self decodeForKey:@"activation"] doubleValue];
        _learningRate     = [[self decodeForKey:@"learningRate"] doubleValue];
        _randomMax        = [[self decodeForKey:@"randomMax"] doubleValue];
        _randomMin        = [[self decodeForKey:@"randomMin"] doubleValue];
        _convergenceError = [[self decodeForKey:@"convergenceError"] integerValue];
        _maxIteration     = [[self decodeForKey:@"maxIteration"] integerValue];
        _iteration        = 0;
        _timestepSize     = [[self decodeForKey:@"batchSize"] integerValue];
        _cost             = [[RNNCost alloc] init];
    }
    return self;
}

@end
